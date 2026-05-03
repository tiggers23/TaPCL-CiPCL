import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix



class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self._correct_t2v = 0
        self._correct_v2t = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred_t2v = []
        self._y_pred_v2t = []
   
        self.r1t =[]
        self.r5t =[]
        self.r10t=[]
        self.medrt=[]
        self.meanrt=[]
        self.r1v=[]
        self.r5v=[]
        self.r10v=[]
        self.medrv=[]
        self.meanrv=[]
        self.sum=[]

    def reset(self):
        self._correct_t2v = 0
        self._correct_v2t = 0
        self._total = 0
        self._y_true = []
        self._y_pred_t2v = []
        self._y_pred_v2t = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def get_gt(self, video_ids, caption_ids, tasknb):
        v2t_gt = []
        num = len(video_ids)
        n = num
        for j, vid_id in enumerate(video_ids):
            if j > n:
                break
            im_id = vid_id.split('/')[-1]
            im_id = im_id.split('.')[0]
            v2t_gt.append([])
            for i, cap_id in enumerate(caption_ids):
                if cap_id.split('#', 1)[0] == im_id:
                    v2t_gt[-1].append(i)
        t2v_gt = {}
        for i, t_gts in enumerate(v2t_gt):
            if i > n:
                break
            for t_gt in t_gts:
                t2v_gt.setdefault(t_gt, [])
                t2v_gt[t_gt].append(i)
        return v2t_gt, t2v_gt
    
    def eval_q2m(self, scores, q2m_gts, tasknb, t2v=False):

        n_q, n_m = scores.shape
        scores = scores.cpu()
        n = int(n_q)
        if t2v:
            gt_ranks = np.zeros((n_q,), np.int32)
            for i in range(n_q):
                s = scores[i][:n]
                sorted_idxs = np.argsort(s)
                rank = n_m + 1
                for k in q2m_gts[i]:
                    tmp = np.where(sorted_idxs == k)[0][0] + 1
                    if tmp < rank:
                        rank = tmp

                gt_ranks[i] = rank
            r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
            r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
            r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
            medr = np.median(gt_ranks)
            meanr = gt_ranks.mean()
        else:
            gt_ranks = np.zeros((n,), np.int32)
            for i in range(n):
                s = scores[i]
                sorted_idxs = np.argsort(s)
                rank = n_m + 1
                for k in q2m_gts[i]:
                    tmp = np.where(sorted_idxs == k)[0][0] + 1
                    if tmp < rank:
                        rank = tmp

                gt_ranks[i] = rank
        # compute metrics
            r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n
            r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n
            r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n
            medr = np.median(gt_ranks)
            meanr = gt_ranks.mean()
        # mAP = aps.mean()

        return r1, r5, r10, medr, meanr

    def process(self, mo, impath, cap_id, tasknb):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        '''pred_t2v = mo.max(1)[1]
        pred_v2t = mo.max(0)[1]
        gt = torch.tensor([i for i in range(mo.shape[0])], dtype=torch.float).to("cuda:2")

        matches_t2v = pred_t2v.eq(gt).float()
        matches_v2t = pred_v2t.eq(gt).float()
        
        self._correct_t2v += int(matches_t2v.sum().item())
        self._correct_v2t += int(matches_v2t.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred_t2v.extend(pred_t2v.data.cpu().numpy().tolist())
        self._y_pred_v2t.extend(pred_v2t.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches_t2v[i].item())
                self._per_class_res[label].append(matches_i)'''
        scores = -1*mo.t()
        v2t_gt, t2v_gt = self.get_gt(impath, cap_id, tasknb)


        '''self.r1t =[]
        self.r5t =[]
        self.r10t=[]
        self.medrt=[]
        self.meanrt=[]
        self.r1v=[]
        self.r5v=[]
        self.r10v=[]
        self.medrv=[]
        self.meanrv=[]'''

        self.t2v_r1, self.t2v_r5, self.t2v_r10, self.t2v_medr, self.t2v_meanr = self.eval_q2m(scores, t2v_gt, tasknb, t2v=True)
        self.v2t_r1, self.v2t_r5, self.v2t_r10, self.v2t_medr, self.v2t_meanr = self.eval_q2m(scores.T, v2t_gt, tasknb)
        '''self.r1t.append(self.t2v_r1)
        self.r5t.append(self.t2v_r5)
        self.r10t.append(self.t2v_r10)
        self.medrt.append(self.t2v_medr)
        self.meanrt.append(self.t2v_meanr)
        self.r1v.append(self.v2t_r1)
        self.r5v.append(self.v2t_r5)
        self.r10v.append(self.v2t_r10)
        self.medrv.append(self.v2t_medr)
        self.meanrv.append(self.v2t_meanr)'''
        self.sum.append(self.t2v_r1)
        self.sum.append(self.t2v_r5)
        self.sum.append(self.t2v_r10)
        self.sum.append(self.v2t_r1)
        self.sum.append(self.v2t_r5)
        self.sum.append(self.v2t_r10)


    def evaluate(self, task_id):
        '''results = OrderedDict()
        acc_t2v = 100.0 * self._correct_t2v / self._total
        acc_v2t = 100.0 * self._correct_v2t / self._total
        err_t2v = 100.0 - acc_t2v
        err_v2t = 100.0 - acc_v2t
        macro_f1_t2v = 100.0 * f1_score(
            self._y_true,
            self._y_pred_t2v,
            average="macro",
            labels=np.unique(self._y_true)
        )
        macro_f1_v2t = 100.0 * f1_score(
            self._y_true,
            self._y_pred_v2t,
            average="macro",
            labels=np.unique(self._y_true)
        )
        # The first value will be returned by trainer.test()
        results["accuracy_t2v"] = acc_t2v
        results["accuracy_v2t"] = acc_v2t
        results["error_rate_t2v"] = err_t2v
        results["error_rate_v2t"] = err_v2t
        results["macro_f1_t2v"] = macro_f1_t2v
        results["macro_f1_v2t"] = macro_f1_v2t

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct_t2v: {self._correct_t2v:,}\n"
            f"* accuracy_t2v: {acc_t2v:.1f}%\n"
            f"* error_t2v: {err_t2v:.1f}%\n"
            f"* macro_f1_t2v: {macro_f1_t2v:.1f}%\n"
            f"* correct_v2t: {self._correct_v2t:,}\n"
            f"* accuracy_v2t: {acc_v2t:.1f}%\n"
            f"* error_v2t: {err_v2t:.1f}%\n"
            f"* macro_f1_v2t: {macro_f1_v2t:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results'''
        results = OrderedDict()
        t2v_r1 = self.t2v_r1
        t2v_r5 = self.t2v_r5
        t2v_r10 = self.t2v_r10
        t2v_medr = self.t2v_medr
        t2v_meanr = self.t2v_meanr
        v2t_r1 = self.v2t_r1
        v2t_r5 = self.v2t_r5
        v2t_r10 = self.v2t_r10
        v2t_medr = self.v2t_medr
        v2t_meanr = self.v2t_meanr
        avg = np.mean(self.sum)

        results[task_id] = {}
        results[task_id]['avg'] = avg
        results[task_id]["t2v_r1"] = t2v_r1
        results[task_id]["t2v_r5"] = t2v_r5
        results[task_id]["t2v_r10"] = t2v_r10
        results[task_id]["t2v_medr"] = t2v_medr
        results[task_id]["t2v_meanr"] = t2v_meanr
        results[task_id]["v2t_r1"] = v2t_r1
        results[task_id]["v2t_r5"] = v2t_r5
        results[task_id]["v2t_r10"] = v2t_r10
        results[task_id]["v2t_medr"] = v2t_medr
        results[task_id]["v2t_meanr"] = v2t_meanr

        '''self.r1t =[]
        self.r5t =[]
        self.r10t=[]
        self.medrt=[]
        self.meanrt=[]
        self.r1v=[]
        self.r5v=[]
        self.r10v=[]
        self.medrv=[]
        self.meanrv=[]'''
        self.sum=[]

        print(
            "=> result\n"
            f"* t2v_r1: {t2v_r1:,}%\n"
            f"* t2v_r5: {t2v_r5:.1f}%\n"
            f"* t2v_r10: {t2v_r10:.1f}%\n"
            f"* t2v_medr: {t2v_medr:.1f}%\n"
            f"* t2v_meanr: {t2v_meanr:.1f}%\n"
            f"* v2t_r1: {v2t_r1:,}%\n"
            f"* v2t_r5: {v2t_r5:.1f}%\n"
            f"* v2t_r10: {v2t_r10:.1f}%\n"
            f"* v2t_medr: {v2t_medr:.1f}%\n"
            f"* v2t_meanr: {v2t_meanr:.1f}%\n"
            f"* avg: {avg:.1f}%"
        )

        return results
