import numpy as np


class _ClsMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """

    def update(self, gt, pred):
        """ Overridden by subclasses """

    def get_results(self):
        """ Overridden by subclasses """

    def to_str(self, metrics):
        """ Overridden by subclasses """

    def reset(self):
        """ Overridden by subclasses """   

class ClsMetrics(_ClsMetrics):

    def __init__(self, n_classes):
        super(ClsMetrics, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        """
        5 0 0 2     acc: scalar
        0 5 0 0     acc_cls: scalar
        0 0 5 0     fwavacc: scalar
        0 0 0 3     iu: iou per class, mean_iu: scalar, cls_iu: K-dim vector
        """
        hist = self.confusion_matrix

        a = np.diag(hist).sum()
        b = hist.sum()
        acc = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        #acc = np.diag(hist).sum() / hist.sum()
        a = np.diag(hist)
        b = hist.sum(axis=1)
        acc_cls = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        #acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        a = np.diag(hist)
        b = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        iu = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        #iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

        a = hist.sum(axis=1)
        b = hist.sum()
        freq = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        #freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        a = np.diag(hist) * 2
        b = (hist.sum(axis=1) + hist.sum(axis=0))
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        #f1 = np.diag(hist) * 2 / (hist.sum(axis=1) + hist.sum(axis=0))
        cls_f1 = dict(zip(range(self.n_classes), f1))
        
        return {
                "Overall Acc": acc, 
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Class F1": cls_f1
            }


if __name__ == "__main__":

    metric = ClsMetrics(n_classes=2)

    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    metric.update(y_true, y_pred)

    print(metric.get_results())
    print(metric.confusion_matrix)