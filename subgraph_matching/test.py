from common import utils
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve
)
import os
import torch
import time
import numpy as np
from tqdm import tqdm

USE_ORCA_FEATS = False # whether to use orca motif counts along with embeddings
MAX_MARGIN_SCORE = 1e9 # a very large margin score to given orca constraints

def validation(args, model, data_source, logger, batch_n, epoch, verbose=False):
    # test on new motifs
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
        train=False, use_distributed_sampling=False)

    start_time = time.time()
    for batch_target, batch_neg_target, batch_neg_query in tqdm(zip(*loaders)):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, train=False)
        if pos_a:
            pos_a = pos_a.to(utils.get_device())
            pos_b = pos_b.to(utils.get_device())
        if neg_a:
            neg_a = neg_a.to(utils.get_device())
            neg_b = neg_b.to(utils.get_device())
        
        labels = torch.tensor([1]*(pos_a.num_graphs if pos_a else 0) +
            [0]*(neg_a.num_graphs if neg_a else 0)).to(utils.get_device())
        with torch.no_grad():
            emb_as = torch.tensor([]).to(utils.get_device())
            emb_bs = torch.tensor([]).to(utils.get_device())
            if pos_a:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                    model.emb_model(pos_b))
                emb_as = torch.cat((emb_as, emb_pos_a), dim=0)
                emb_bs = torch.cat((emb_bs, emb_pos_b), dim=0)
            if neg_a:
                emb_neg_a, emb_neg_b = (model.emb_model(neg_a),
                                        model.emb_model(neg_b))
                emb_as = torch.cat((emb_as, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_bs, emb_neg_b), dim=0)
            pred = model(emb_as, emb_bs)
    
            raw_pred = model.predict(pred)
            if USE_ORCA_FEATS:
                import orca
                import matplotlib.pyplot as plt
                def make_feats(g):
                    counts5 = np.array(orca.orbit_counts("node", 5, g))
                    for v, n in zip(counts5, g.nodes):
                        if g.nodes[n]["node_feature"][0] > 0:
                            anchor_v = v
                            break
                    v5 = np.sum(counts5, axis=0)
                    return v5, anchor_v
                for i, (ga, gb) in enumerate(zip(neg_a.G, neg_b.G)):
                    (va, na), (vb, nb) = make_feats(ga), make_feats(gb)
                    if (va < vb).any() or (na < nb).any():
                        raw_pred[pos_a.num_graphs + i] = MAX_MARGIN_SCORE

            if args.method_type == "order":
                pred = model.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
                raw_pred *= -1
            elif args.method_type == "ensemble":
                pred = torch.stack([m.clf_model(
                    raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
                for i in range(pred.shape[1]):
                    print(pred[:,i])
                pred = torch.min(pred, dim=0)[0]
                raw_pred *= -1
            elif args.method_type == "mlp":
                raw_pred = raw_pred[:,1]
                pred = pred.argmax(dim=-1)
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    end_time = time.time()
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    test_time = (end_time - start_time) / len(labels)
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    result_rows = []
    
    if args.test:
        for conf_step in [
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.91,
            0.92,
            0.93,
            0.94,
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
        ]:
            test_pred_by_conf = raw_pred.copy()
            test_pred_by_conf[test_pred_by_conf < conf_step] = 0
            test_pred_by_conf[test_pred_by_conf > 0] = 1
        
            test_roc = roc_auc_score(labels, test_pred_by_conf)
            test_acc = accuracy_score(labels, test_pred_by_conf)
            test_pre = precision_score(labels, test_pred_by_conf)
            test_rec = recall_score(labels, test_pred_by_conf)
            test_f1s = f1_score(labels, test_pred_by_conf)
            test_prc = average_precision_score(labels, test_pred_by_conf)
            
            result_rows.append(
                    [
                        conf_step,
                        test_time,
                        test_roc,
                        test_prc,
                        test_pre,
                        test_rec,
                        test_f1s,
                        test_acc,
                    ]
                )
            
        result_file = args.dataset.split("/")[-1] + ".csv"
        with open(os.path.join(args.result_dir, result_file), "w", encoding="utf-8") as f:
            f.write(
                "Confident,Execution Time,ROC AUC,PR AUC,Precision,Recall,F1-Score,Accuracy\n"
            )
            for row in result_rows:
                f.write(",".join([str(x) for x in row]))
                f.write("\n")

    else:
        test_roc = roc_auc_score(labels, pred)
        test_acc = accuracy_score(labels, pred)
        test_f1s = f1_score(labels, pred)
        test_prc = average_precision_score(labels, pred)
        
        print("\n{}".format(str(datetime.now())))
        print("Validation. Epoch {}. Time: {:.5f}. ROC AUC: {:.5f}. PR AUC: {:.5f}. F1-Score: {:.5f}. Accuracy: {:.5f}".format(
            epoch, test_time, test_roc, test_prc, test_f1s, test_acc))
        
        logger.add_scalar("Accuracy/test", test_acc, batch_n)
        logger.add_scalar("F1 Score/test", test_f1s, batch_n)
        logger.add_scalar("AUROC/test", test_roc, batch_n)
        logger.add_scalar("AvgPrec/test", test_prc, batch_n)
        print("Saving {}".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)
    
    if verbose:
        import matplotlib.pyplot as plt
        precs, recalls, threshs = precision_recall_curve(labels, pred)
        plt.plot(recalls, precs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("plots/precision-recall-curve.png")
        print("Saved PR curve plot in plots/precision-recall-curve.png")

    # if verbose:
    #     conf_mat_examples = defaultdict(list)
    #     idx = 0
    #     for batch_target, batch_neg_target, batch_neg_query in tqdm(zip(*loaders)):
    #         pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
    #             batch_neg_target, batch_neg_query, train=False)
    #         if pos_a:
    #             pos_a = pos_a.to(utils.get_device())
    #             pos_b = pos_b.to(utils.get_device())
    #         if neg_a:
    #             neg_a = neg_a.to(utils.get_device())
    #             neg_b = neg_b.to(utils.get_device())
    #         for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
    #             if not list_a: continue
    #             for a, b in zip(list_a.G, list_b.G):
    #                 correct = pred[idx] == labels[idx]
    #                 conf_mat_examples[correct, pred[idx]].append((a, b))
    #                 idx += 1

if __name__ == "__main__":
    from subgraph_matching.train import main
    main(force_test=True)
