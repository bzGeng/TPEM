import json
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
import shutil
from models.modules import *
from models.layers import *
from utils import Optimizers, get_optimize_parameters
from utils.masked_cross_entropy import *
from utils.measures import moses_multi_bleu


class TPEM(nn.Module):
    def __init__(self, hidden_size, lang, langs, max_resp_len, task, free_ratio, lr, n_layers, dropout, task_history, hidden_sizes,
                 growing_embeddings, growing_embeddings_masks, tasks_specific_bias, tasks_specific_decoders, mode):
        super(TPEM, self).__init__()
        self.name = "TPEM"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.langs = langs
        self.free_ratio = free_ratio
        self.lr = lr
        self.mode = mode
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)
        self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
        self.extKnow = ExternalKnowledge(lang.n_words, hidden_size, n_layers, dropout)
        self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang, hidden_size, self.decoder_hop,
                                          dropout)

        self.growing_embeddings = growing_embeddings
        self.growing_embeddings_masks = growing_embeddings_masks
        self.tasks_specific_bias = tasks_specific_bias
        self.tasks_specific_decoders = tasks_specific_decoders
        self.tasks = task_history
        self.hidden_sizes = hidden_sizes
        if task not in self.tasks:
            self.tasks.append(task)
            self.langs[self.task] = lang
            self.hidden_sizes.append(hidden_size)

        self.weight_decay = 5e-5
        self.first_prune_percentage = 0.5
        if mode == 'prune' or mode == 'test':
            self.current_task_idx = self.tasks.index(self.task) + 1
        elif mode == 'train':
            self.current_task_idx = len(self.tasks) - 1

        self.masks = None
        self.piggymasks = {}
        self.piggymask_optimizer = None
        self.piggymask_scheduler = None
        self.inference_task_idx = self.tasks.index(task) + 1

        self.build_optimizers_and_scheduler(lr)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()

    def build_optimizers_and_scheduler(self, lr):
        self.encoder_optimizer = optim.Adam(get_optimize_parameters(self.encoder.named_parameters()), lr=lr)
        self.extKnow_optimizer = optim.Adam(get_optimize_parameters(self.extKnow.named_parameters()), lr=lr)
        self.decoder_optimizer = optim.Adam(get_optimize_parameters(self.decoder.named_parameters()), lr=lr)
        self.optimizers = Optimizers()
        self.optimizers.add(self.encoder_optimizer, lr)
        self.optimizers.add(self.extKnow_optimizer, lr)
        self.optimizers.add(self.decoder_optimizer, lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5,
                                                        patience=1, min_lr=1e-4, verbose=True)

    def load_checkpoint(self, path):
        if path is not None:
            print("MODEL {} LOADED".format(str(path)))
            curr_encoder = self.encoder.state_dict()
            curr_extKnow = self.extKnow.state_dict()
            curr_decoder = self.decoder.state_dict()
            checkpoint_encoder = torch.load(str(path) + '/enc.th')
            checkpoint_extKnow = torch.load(str(path) + '/enc_kb.th')
            checkpoint_decoder = torch.load(str(path) + '/dec.th')
            module_current = [curr_encoder, curr_extKnow, curr_decoder]
            module_checkpoint = [checkpoint_encoder, checkpoint_extKnow, checkpoint_decoder]
            for idx, module in enumerate(module_checkpoint):
                for name, param in module.items():
                    if 'piggy' in name:
                        continue
                    if 'emb' in name or 'C' in name:
                        if self.mode == 'train':
                            continue
                    else:
                        if len(param.size()) == 2:
                            module_current[idx][name][:param.size(0), :param.size(1)].copy_(
                                    param[:param.size(0), :param.size(1)])
                        elif len(param.size()) == 1:
                            module_current[idx][name][:param.size(0)].copy_(param[:param.size(0)])
            self.load_from_growing_embeddings()

    def load_from_growing_embeddings(self):
        module_list = ['encoder', 'extKnow']
        count = 0
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    for key, value in self.lang.word2index.items():
                        if key in self.growing_embeddings[superior_module][name]:
                            pre_emb_max_size = self.growing_embeddings_masks[superior_module][name][key].size(0)
                            module.state_dict()['weight'][value][:pre_emb_max_size].copy_(
                                self.growing_embeddings[superior_module][name][key].data[:pre_emb_max_size])
                            count += 1
                    print("{} words embeddings reloaded !".format(count))
                    count = 0

    def load_from_growing_embeddings_for_infer(self):
        module_list = ['encoder', 'extKnow']
        count = 0
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    for key, value in self.lang.word2index.items():
                        if key in self.growing_embeddings[superior_module][name]:
                            module.state_dict()['weight'][value].copy_(
                                self.growing_embeddings[superior_module][name][key].data[:self.hidden_size])
                            count += 1
                    print("{} words embeddings reloaded !".format(count))
                    count = 0

    def load_decoder_from_previous(self):
        print("DECODER FOR {} LOADED".format(str(self.task)))
        curr_decoder = self.decoder.state_dict()
        checkpoint_decoder = self.tasks_specific_decoders[self.task]
        for name, param in checkpoint_decoder.items():
            if len(param.size()) == 2:
                curr_decoder[name][:param.size(0), :param.size(1)].copy_(param)
            elif len(param.size()) == 1:
                curr_decoder[name][:param.size(0)].copy_(param)

    def load_checkpoint_for_infer(self, path):
        if path is not None:
            print("MODEL {} LOADED".format(str(path)))
            curr_encoder = self.encoder.state_dict()
            curr_extKnow = self.extKnow.state_dict()
            checkpoint_encoder = torch.load(str(path) + '/enc.th')
            checkpoint_extKnow = torch.load(str(path) + '/enc_kb.th')
            growing_embeddings = torch.load(path + '/continual_learning_info.th')['growing_embeddings']
            growing_embeddings_masks = torch.load(path + '/continual_learning_info.th')['growing_embeddings_masks']
            self.growing_embeddings = growing_embeddings
            self.growing_embeddings_masks = growing_embeddings_masks
            module_current = [curr_encoder, curr_extKnow]
            module_checkpoint = [checkpoint_encoder, checkpoint_extKnow]
            for idx, module in enumerate(module_checkpoint):
                for name, param in module.items():
                    if 'piggy' in name:
                        continue
                    if 'emb' in name or 'C' in name:
                        if self.mode == 'train' or self.mode == 'test':
                            continue
                        module_current[idx][name][:param.size(0), :param.size(1)].copy_(param)
                    else:
                        if len(param.size()) == 2:
                            dim1, dim2 = module_current[idx][name].size()
                            module_current[idx][name].copy_(param[:dim1, :dim2])
                        elif len(param.size()) == 1:
                            dim = module_current[idx][name].size(0)
                            module_current[idx][name].copy_(param[:dim])
            self.load_from_growing_embeddings_for_infer()
            self.load_task_specific_bias()

    def load_task_specific_bias(self):
        current_task_specific_bias = self.tasks_specific_bias[self.task]
        module_list = ['encoder']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear):
                    module.bias = current_task_specific_bias[superior_module][name]
                if isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'bias' in gru_weight_name:
                            module.state_dict()[gru_weight_name].copy_(
                                current_task_specific_bias[superior_module][name][gru_weight_name].data)

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LG:{:.2f},LV:{:.2f},LL:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v,
                                                                   print_loss_l)

    def calculate_curr_task_ratio(self):
        total_elem = 0
        curr_task_elem = 0

        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                    mask = self.masks[superior_module][name]
                    total_elem += mask.numel()
                    curr_task_elem += torch.sum(mask.eq(self.inference_task_idx))
                elif isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            total_elem += mask.numel()
                            curr_task_elem += torch.sum(mask.eq(self.inference_task_idx))
        return float(curr_task_elem.cpu()) / total_elem

    def calculate_shared_part_ratio(self):
        total_elem = 0
        shared_elem = 0

        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                    mask = self.masks[superior_module][name]
                    self_total_elem = torch.sum(mask.gt(0) & mask.lt(self.inference_task_idx))
                    self_shared_elem = torch.sum(torch.where(
                        mask.gt(0).cuda() & mask.lt(self.inference_task_idx).cuda() & module.piggymask.gt(0.005).cuda(),
                        torch.tensor(1).cuda(), torch.tensor(0).cuda()))
                    self_masked_elem = self_total_elem - self_shared_elem
                    self_masked_ratio = self_masked_elem.float() / self_total_elem
                    total_elem += self_total_elem
                    shared_elem += self_shared_elem
                    print("{}.{}: masked_elem {}, masked_ratio {:.4f}".format(superior_module, name, self_masked_elem, self_masked_ratio))
                elif isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            self_total_elem = torch.sum(mask.gt(0) & mask.lt(self.inference_task_idx))
                            self_shared_elem = torch.sum(
                                torch.where(mask.gt(0).cuda() & mask.lt(self.inference_task_idx).cuda() &
                                            module.piggymasks_weights[gru_weight_name + '_piggymask'].gt(0.005).cuda(),
                                            torch.tensor(1).cuda(), torch.tensor(0).cuda()))
                            self_masked_elem = self_total_elem - self_shared_elem
                            self_masked_ratio = self_masked_elem.float() / self_total_elem
                            total_elem += self_total_elem
                            shared_elem += self_shared_elem
                            print("{}.{}.{}: masked_elem {}, masked_ratio {:.4f}".format(superior_module, name, gru_weight_name, self_masked_elem, self_masked_ratio))
        if total_elem.cpu() != 0.0:
            return float(shared_elem.cpu()) / float(total_elem.cpu())
        else:
            return 0.0

    def save_model(self, dec_type, epoch=None):
        layer_info = str(self.n_layers)
        epoch_str = ('EPOCH' + str(epoch)) if epoch else ""
        mode_str = 'save_train/' if self.mode == 'train' else 'save_prune/'
        directory = mode_str + str(self.task) + '/' + epoch_str + '/' + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(
            self.lr) + str(dec_type)
        file_path = mode_str + str(self.task)
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder.state_dict(), directory + '/enc.th')
        torch.save(self.extKnow.state_dict(), directory + '/enc_kb.th')
        torch.save(self.decoder.state_dict(), directory + '/dec.th')
        torch.save(self.lang, directory + '/lang.th')

        current_growing_embeddings = self.update_growing_embeddings()
        current_growing_embeddings_masks = self.update_growing_embeddings_masks()
        self.update_tasks_specific_bias()
        self.update_tasks_specific_decoder()


        continual_learning_info = {
            'masks': self.masks,
            'task_history': self.tasks,
            'langs': self.langs,
            'growing_embeddings': current_growing_embeddings,
            'growing_embeddings_masks': current_growing_embeddings_masks,
            'tasks_specific_bias': self.tasks_specific_bias,
            'tasks_specific_decoders': self.tasks_specific_decoders,
            'piggymasks': self.piggymasks,
            'hidden_sizes': self.hidden_sizes,
            'free_ratio': self.free_ratio
        }
        torch.save(continual_learning_info, directory + '/continual_learning_info.th')

    def update_growing_embeddings(self):
        current_growing_embeddings = self.growing_embeddings
        module_list = ['encoder', 'extKnow']
        if len(self.growing_embeddings) == 0:
            for superior_module in module_list:
                current_growing_embeddings[superior_module] = {}
                named_modules = getattr(self, superior_module).named_modules()
                for name, module in named_modules:
                    if isinstance(module, SharableEmbedding):
                        current_growing_embeddings[superior_module][name] = {}
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    for key, value in self.lang.word2index.items():
                        current_growing_embeddings[superior_module][name][key] = module.weight[value]
        return current_growing_embeddings

    def update_growing_embeddings_masks(self):
        current_growing_embeddings_masks = self.growing_embeddings_masks
        module_list = ['encoder', 'extKnow']
        if len(self.growing_embeddings_masks) == 0:
            for superior_module in module_list:
                current_growing_embeddings_masks[superior_module] = {}
                named_modules = getattr(self, superior_module).named_modules()
                for name, module in named_modules:
                    if isinstance(module, SharableEmbedding):
                        current_growing_embeddings_masks[superior_module][name] = {}
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    for key, value in self.lang.word2index.items():
                        current_growing_embeddings_masks[superior_module][name][key] = \
                            self.masks[superior_module][name][value]
        return current_growing_embeddings_masks

    def update_tasks_specific_bias(self):
        if self.task not in self.tasks_specific_bias:
            self.tasks_specific_bias[self.task] = {}
        module_list = ['encoder']
        current_task_specific_bias = self.tasks_specific_bias[self.task]
        if len(current_task_specific_bias) == 0:
            for superior_module in module_list:
                current_task_specific_bias[superior_module] = {}
                named_modules = getattr(self, superior_module).named_modules()
                for name, module in named_modules:
                    if isinstance(module, SharableLinear):
                        current_task_specific_bias[superior_module][name] = {}
                    if isinstance(module, SharableGRU):
                        current_task_specific_bias[superior_module][name] = {}
                        for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                            if 'bias' in gru_weight_name:
                                current_task_specific_bias[superior_module][name][gru_weight_name] = {}
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear):
                    current_task_specific_bias[superior_module][name] = module.bias
                if isinstance(module, SharableGRU):
                    current_task_specific_bias[superior_module][name] = {}
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'bias' in gru_weight_name:
                            current_task_specific_bias[superior_module][name][gru_weight_name] = getattr(module,
                                                                                                         gru_weight_name)
        self.tasks_specific_bias[self.task] = current_task_specific_bias

    def update_tasks_specific_decoder(self):
        task_specific_decoder = self.decoder.state_dict()
        del task_specific_decoder['C.weight']
        if 'C.piggymask' in task_specific_decoder:
            del task_specific_decoder['C.piggymask']
        self.tasks_specific_decoders[self.task] = task_specific_decoder

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l = 0, 1, 0, 0, 0

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizers.zero_grad()
        if self.piggymask_optimizer is not None:
            self.piggymask_optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _, global_pointer = \
            self.encode_and_decode(data, max_target_length, use_teacher_forcing, False)

        # Loss calculation and backpropagation
        loss_g = self.criterion_bce(global_pointer, data['selector_index'])
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
            data['sketch_response'].contiguous(),
            data['response_lengths'])
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),
            data['ptr_index'].contiguous(),
            data['response_lengths'])
        loss = loss_g + loss_v + loss_l

        loss.backward()

        self.do_weight_decay_and_make_grads_zero()

        if self.piggymask_optimizer is not None:
            self.piggymask_optimizer.step()
            self.piggymask_scheduler.step()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Clean the gradient of PAD token
        self.clean_pad_grad()

        # Update parameters with optimizers
        self.optimizers.step()

        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()

    def clean_pad_grad(self):
        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    module.weight.grad[module.padding_idx] = 0

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):
        # Build unknown mask for memory
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
                conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
            rand_mask = self._cuda(rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
        else:
            story, conv_story = data['context_arr'], data['conv_arr']

        # Encode dialog history and KB to vectors
        dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])
        global_pointer, kb_readout = self.extKnow.load_memory(story, data['kb_arr_lengths'], data['conv_arr_lengths'],
                                                              dh_hidden, dh_outputs)
        encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout), dim=1)

        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [word_arr[0] for word_arr in elm]
            self.copy_list.append(elm_temp)

        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            self.extKnow,
            story.size(),
            data['context_arr_lengths'],
            self.copy_list,
            encoded_hidden,
            data['sketch_response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words,
            global_pointer
        )

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer

    def evaluate(self, dev, matric_best, epoch=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)

        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev), total=len(dev))

        if args['dataset'] == 'kvr':
            entity_path = 'data/KVR/kvret_entities.json'
        elif args['dataset'] == 'cam':
            entity_path = 'data/CamRest/camrest676-entities.json'
        elif args['dataset'] == 'woz':
            entity_path = 'data/MULTIWOZ2.1/global_entities.json'

        with open(entity_path) as f:
            global_entity = json.load(f)
            global_entity_list = []
            for key in global_entity.keys():
                if key != 'poi':
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                else:
                    for item in global_entity['poi']:
                        global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]

            global_entity_list = list(set(global_entity_list))
        for j, data_dev in pbar:
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse, global_pointer = self.encode_and_decode(data_dev, self.max_resp_len,
                                                                                        False, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)

                single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                    global_entity_list, data_dev['kb_arr_plain'][bi])
                F1_pred += single_f1
                F1_count += count

                # compute Per-response Accuracy Score
                total += 1
                if gold_sent == pred_sent:
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t" + str(acc_score))

        F1_score = F1_pred / float(F1_count)
        print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
        print("BLEU SCORE:\t" + str(bleu_score))

        mix_score = bleu_score / 25 + F1_score / 0.5
        if mix_score >= matric_best:
            self.save_model('ENTF1-{:.4f}'.format(F1_score), epoch)
            print("MODEL SAVED")
        return mix_score, bleu_score, F1_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx]) - data['conv_arr_lengths'][batch_idx] - 1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w != 'PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1] == flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr, ': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')

    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.masks
        self.current_task_idx += 1

        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                    mask = self.masks[superior_module][name]
                    mask[mask.eq(0)] = self.current_task_idx
                if isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            mask[mask.eq(0)] = self.current_task_idx

    def do_weight_decay_and_make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.masks
        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                    mask = self.masks[superior_module][name]
                    if module.weight.grad is not None:
                        module.weight.grad.data.add_(self.weight_decay, module.weight.data)
                        module.weight.grad.data[mask.ne(self.current_task_idx)] = 0
                    if module.piggymask is not None and module.piggymask.grad is not None:
                        if self.mode == 'prune':
                            module.piggymask.grad.data.fill_(0)
                if isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            getattr(module, gru_weight_name).grad.data.add_(self.weight_decay,
                                                                            getattr(module, gru_weight_name).data)
                            getattr(module, gru_weight_name).grad.data[mask.ne(self.current_task_idx)] = 0
                            if module.piggymasks_weights[gru_weight_name + '_piggymask'] is not None \
                                    and module.piggymasks_weights[gru_weight_name + '_piggymask'].grad is not None:
                                if self.mode == 'prune':
                                    module.piggymasks_weights[gru_weight_name + '_piggymask'].grad.data.fill_(0)

        for name, module in self.decoder.named_modules():
            if isinstance(module, SharableLinear):
                mask = self.masks['decoder'][name]
                if module.weight.grad is not None:
                    module.weight.grad.data[mask.ne(self.current_task_idx)] = 0
            if isinstance(module, SharableGRU):
                for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                    if 'weight' in gru_weight_name:
                        mask = self.masks['decoder'][name][gru_weight_name]
                        getattr(module, gru_weight_name).grad.data[mask.ne(self.current_task_idx)] = 0

    def one_shot_prune(self, weight, mask, one_shot_prune_percentage):
        tensor = weight[mask.eq(self.current_task_idx) | mask.eq(0)]  # This will flatten weights
        abs_tensor = tensor.abs()
        cutoff_rank = round(one_shot_prune_percentage * tensor.numel())
        cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].cuda()  # value at cutoff rank

        remove_mask = weight.abs().le(cutoff_value) * mask.eq(self.current_task_idx).cuda()
        # mask = 1 - remove_mask
        mask[remove_mask.eq(1)] = 0

    def row_one_shot_prune(self, weight, mask, one_shot_prune_percentage):
        for weight_row, mask_row in zip(weight, mask):
            if mask_row.lt(self.current_task_idx).sum() > 0:
                one_shot_prune_percentage = one_shot_prune_percentage
            else:
                one_shot_prune_percentage = self.first_prune_percentage
            tensor = weight_row[mask_row.eq(self.current_task_idx) | mask_row.eq(0)]
            abs_tensor = tensor.abs()
            cutoff_rank = round(one_shot_prune_percentage * tensor.numel())
            if cutoff_rank == 0:
                continue
            cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].cuda()  # value at cutoff rank
            remove_mask = weight_row.abs().le(cutoff_value.cuda()) * mask_row.eq(self.current_task_idx).cuda()
            # mask = 1 - remove_mask
            mask_row[remove_mask.eq(1)] = 0

    def do_one_shot_prune(self, one_shot_prune_percentage):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % self.current_task_idx)
        print('Pruning each layer by removing %.2f%% of values' % (100 * one_shot_prune_percentage))

        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear):
                    mask = self.masks[superior_module][name]
                    self.one_shot_prune(module.weight, mask, one_shot_prune_percentage)
                elif isinstance(module, SharableEmbedding):
                    mask = self.masks[superior_module][name]
                    self.row_one_shot_prune(module.weight, mask, one_shot_prune_percentage)
                elif isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            self.one_shot_prune(getattr(module, gru_weight_name), mask, one_shot_prune_percentage)
        return

    def adjust_mask(self):
        """Sets grads of fixed weights to 0."""
        assert self.masks
        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    mask = self.masks[superior_module][name]
                    if module.weight.size(0) != mask.size(0) or module.weight.size(1) != mask.size(1):
                        new_mask = torch.ByteTensor(module.weight.data.size()).fill_(self.current_task_idx)
                        new_mask[:mask.size(0), :mask.size(1)].copy_(mask)
                        if 'cuda' in module.weight.data.type():
                            new_mask = new_mask.cuda()
                        self.masks[superior_module][name] = new_mask
                elif isinstance(module, SharableLinear):
                    mask = self.masks[superior_module][name]
                    if module.weight.size(1) != mask.size(1):
                        new_mask = torch.ByteTensor(module.weight.data.size()).fill_(self.current_task_idx)
                        new_mask[:mask.size(0), :mask.size(1)].copy_(mask)
                        if 'cuda' in module.weight.data.type():
                            new_mask = new_mask.cuda()
                        self.masks[superior_module][name] = new_mask
                elif isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            if getattr(module, gru_weight_name).data.size(1) != mask.size(1):
                                new_mask = torch.ByteTensor(getattr(module, gru_weight_name).data.size()).fill_(
                                    self.current_task_idx)
                                new_mask[:mask.size(0), :mask.size(1)].copy_(mask)
                                if 'cuda' in getattr(module, gru_weight_name).data.type():
                                    new_mask = new_mask.cuda()
                                self.masks[superior_module][name][gru_weight_name] = new_mask

    def apply_mask(self):
        module_list = ['encoder', 'extKnow']
        for superior_module in module_list:
            named_modules = getattr(self, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                    if superior_module == 'decoder' and isinstance(module, SharableEmbedding):
                        continue
                    mask = self.masks[superior_module][name].cuda()
                    module.weight.data[mask.eq(0)] = 0.0
                    module.weight.data[mask.gt(self.inference_task_idx)] = 0.0
                if isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = self.masks[superior_module][name][gru_weight_name]
                            getattr(module, gru_weight_name).data[mask.eq(0)] = 0.0
                            getattr(module, gru_weight_name).data[mask.gt(self.inference_task_idx)] = 0.0
