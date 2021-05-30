from tqdm import tqdm
from utils.utils_general import *
from utils.config import *
from models.GLMP import *
from utils import check_resume_folder, info_reload

set_seeds(args["seed"])
torch.set_num_threads(1)
full_path = None
if args["path"] is not None:
    full_path = check_resume_folder(args["path"])

# Configure models and load data
train, dev, test, testOOV, lang = prepare_data_seq_for_domain_task(args['task'], int(args['batch']))

task_history, hidden_sizes, masks, tasks_specific_bias, tasks_specific_decoders, \
langs, growing_embeddings, growing_embeddings_masks, free_ratio = info_reload(full_path)

model = globals()[args['decoder']](
    int(args['hidden']),
    lang,
    langs,
    100,
    args['task'],
    free_ratio,
    lr=float(args['learn']),
    n_layers=int(args['layer']),
    dropout=float(args['drop']),
    task_history=task_history,
    hidden_sizes=hidden_sizes,
    growing_embeddings=growing_embeddings,
    growing_embeddings_masks=growing_embeddings_masks,
    tasks_specific_bias=tasks_specific_bias,
    tasks_specific_decoders=tasks_specific_decoders,
    mode=args['mode']
)

model.load_checkpoint(full_path)


def training_process(model):
    avg_best, cnt, acc = 0.0, 0, 0.0
    max_epoch = 200
    for epoch in range(max_epoch):
        print("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data, int(args['clip']), reset=(i == 0))
            pbar.set_description(model.print_loss())
            # break
        if (epoch + 1) % int(args['evalp']) == 0:
            acc, _, _ = model.evaluate(dev, avg_best, epoch + 1)
            model.scheduler.step(acc)
            if acc >= avg_best:
                avg_best = acc
                cnt = 0
            else:
                cnt += 1

            if cnt == 8:
                print("Ran out of patient, early stop...")
                break


training_process(model)
