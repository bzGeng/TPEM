from tqdm import tqdm
from utils.utils_general import *
from utils.config import *
import math
from models.TPEM import *
from utils import check_resume_folder, load_or_build_masks, load_or_build_piggymasks, info_reload

set_seeds(args["seed"])
torch.set_num_threads(1)
full_path = None
if args["path"] is not None:
    full_path = check_resume_folder(args["path"])

# Configure models and load data
train, dev, test, testOOV, lang = prepare_data_seq_for_domain_task(args['task'], int(args['batch']))

task_history, hidden_sizes, masks, tasks_specific_bias, tasks_specific_decoders, \
langs, growing_embeddings, growing_embeddings_masks, free_ratio = info_reload(full_path)

if len(hidden_sizes) == 0:
    pass
elif args['mode'] == 'train':
    free_ratio *= (1 - args['one_shot_prune_percentage'])
    print("Pre free ratio: {:.2f}".format(free_ratio))
    expand_ratio = (args['one_shot_prune_percentage']-free_ratio) * math.log(1+(len(train)/args['beta']))
    print("Expand ratio: {:.2f}".format(expand_ratio))
    args['hidden'] = int(hidden_sizes[-1] + args['alpha'] * expand_ratio)
    free_ratio = free_ratio + (args['alpha'] * expand_ratio / args['hidden'])
    print("Post free ratio: {:.2f}".format(free_ratio))
else:
    args['hidden'] = hidden_sizes[task_history.index(args['task'])]

print("Current hidden size: {}".format(args['hidden']))

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

model.masks = load_or_build_masks(masks, model, growing_embeddings_masks)

model.load_checkpoint(full_path)

if args['mode'] == 'train':
    model.make_finetuning_mask()
    model.adjust_mask()
elif args['mode'] == 'prune':
    print('Sparsity ratio: {}'.format(args['one_shot_prune_percentage']))
    print('Execute one shot pruning ...')
    model.do_one_shot_prune(args['one_shot_prune_percentage'])
    model.apply_mask()

if full_path is not None:
    model.piggymasks = load_or_build_piggymasks(full_path, model, len(train))

curr_task_ratio = model.calculate_curr_task_ratio()
print('Current Task Ratio:{:.2f}'.format(curr_task_ratio))


def training_process(model):
    avg_best, cnt, acc = 0.0, 0, 0.0
    max_epoch = 200 if args['mode'] == 'train' else 6
    for epoch in range(max_epoch):
        print("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        if args['mode'] == 'train' and model.piggymask_optimizer is not None:
            shared_part_ratio = model.calculate_shared_part_ratio()
            print('SPR:{:.2f}'.format(shared_part_ratio))
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
