from tqdm import tqdm
from utils.utils_general import *
from utils.config import *
from models.GLMP import *
from utils import CsvLogger, check_resume_folder, info_reload

full_path = None
if args["path"] is not None:
    full_path = check_resume_folder(args["path"])

directory = full_path.split("/")
task = directory[3].split('HDD')[0]
HDD = directory[3].split('HDD')[1].split('BSZ')[0]
L = directory[3].split('L')[1].split('lr')[0].split("-")[0]
decoder = 'GLMP'
BSZ = int(directory[3].split('BSZ')[1].split('DR')[0])
DS = args["dataset"]

logger = CsvLogger(file_name='GLMP_continual_middle_results', resume=True, path='results', data_format='csv')
train, dev, test, testOOV, lang = prepare_data_seq_for_domain_task(args['task'], BSZ)
task_history, hidden_sizes, masks, tasks_specific_bias, tasks_specific_decoders, \
langs, growing_embeddings, growing_embeddings_masks, free_ratio = info_reload(full_path)
hidden_size = hidden_sizes[task_history.index(args['task'])]
model = globals()[decoder](
    hidden_size,
    langs[args['task']],
    langs,
    100,
    args['task'],
    free_ratio,
    lr=0.0,
    n_layers=int(L),
    dropout=0.0,
    task_history=task_history,
    hidden_sizes=hidden_sizes,
    growing_embeddings=growing_embeddings,
    growing_embeddings_masks=growing_embeddings_masks,
    tasks_specific_bias=tasks_specific_bias,
    tasks_specific_decoders=tasks_specific_decoders,
    mode=args['mode']
)

model.load_checkpoint_for_infer(full_path)
model.load_decoder_from_previous()

_, bleu, F1 = model.evaluate(test, 1e7)
logger.add(task=args['task'], idx=task_history.index(args['task']), finished=len(task_history), bleu=round(bleu, 4), F1=round(F1*100, 4))
logger.save()
if testOOV != []:
    acc_oov_test = model.evaluate(testOOV, 1e7)

