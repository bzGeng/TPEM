# Code and data for the ACL'2021 paper "Continual Learning for Task-oriented Dialogue System with Iterative Network Pruning, Expanding and Masking"
## requirements

- Python >=3.7
- Pytorch 1.2.0

## 1.You can run TPEM with

```bash
$ bash experiment/run_TPEM.sh 
```

**After completing the training process, you can use the following bash to obtain all middle results**

```bash
$ bash experiment/eval_TPEM.sh 
```
##  2.To observe the “catastrophic  forgetting” of base model, you can run
```bash
$ bash experiment/run_GLMP_continual.sh 
```
**Obtain all middle results with**

```bash 
$ bash experiment/eval_GLMP_continual.sh
```
##  3.To run Re-init which need to save all 7 models:
```bash
$ bash experiment/run_GLMP_Re-init.sh 
```
**Obtain all middle results with**

```bash 
$ bash experiment/eval_GLMP_Re-init.sh
```
##  4.Run TPEM with random task order

```bash
$ bash experiment/run_TPEM_with_random_task_order.sh 
```
**To evaluate shuffle order results**

```bash
$ bash experiment/eval_TPEM_with_shuffle_order.sh 
```



If you find our work helpful, you can also refer to 

SIGIR'2021 paper "Iterative Network Pruning with Uncertainty Regularization for Lifelong Sentiment Classification" 

**IPRLS**:   [https://github.com/siat-nlp/IPRLS](https://github.com/siat-nlp/IPRLS)

