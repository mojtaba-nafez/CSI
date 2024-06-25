from common.eval import *
from evals.ood_pre import eval_ood_detection

model.eval()
print(P)

with torch.no_grad():
    auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader,
                                    train_loader=train_loader, simclr_aug=simclr_aug)
if P.one_class_idx is not None:
    mean_dict = dict()
    mean = 0
    for ood in auroc_dict.keys():
        mean += auroc_dict[ood]
    auroc_dict['one_class_mean'] = mean / len(auroc_dict.keys())

for ood in auroc_dict.keys():
    message = ''
    auroc = auroc_dict[ood]
    message += '[%s %.4f] ' % (ood , auroc)
    if P.print_score:
        print(message)
