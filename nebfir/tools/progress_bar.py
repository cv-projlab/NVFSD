import time
import tqdm
from torch.utils.data.dataloader import  DataLoader



def tprint(s: str):
    tqdm.tqdm.write(s)


class PBar(object):
    def __init__(self):
        self.multi_pbar = {}

        self.start = 0
        self.stop = 1000
        self.range = range(self.start,self.stop)
        self.bar_fmt = '{desc}{percentage:3.0f}%|{bar:24}| [{elapsed}<{remaining}]'
        self.position = 0

    
    def create_pbar(self, key, iter_tuple=None, bar_fmt=None, description=''):

        if iter_tuple is None:
            iter1 = self.stop
            aux_iter = self.range
        else:
            if isinstance(iter_tuple, range) or isinstance(iter_tuple, type(iter([0,1,2]))):
                aux_iter = iter_tuple
                # print(aux_iter)
            else:
                iter0, iter1 = iter_tuple
                if isinstance(iter1, int):
                    aux_iter = range(iter0, iter1)
                elif isinstance(iter1, DataLoader):
                    aux_iter = range(iter0, len(iter1))
                else:
                    raise ValueError('Current available options for iter_tuple are int type and DataLoader type')
            

        self.multi_pbar[key] = {
                                'pb':tqdm.tqdm(iterable=aux_iter, 
                                                position=self.position,
                                                bar_format=bar_fmt if bar_fmt is not None else self.bar_fmt,
                                                leave=True),
                                'position':self.position,
                                'description':description,
                                'range':aux_iter,
                                }

        self.set_description_pbar(key, description)

        self.position += 1


    def set_description_pbar(self, key, description):
        self.multi_pbar[key]['pb'].set_description_str(description)


    def update_pbar(self, key, description=''):
        self.set_description_pbar(key, description)
        self.multi_pbar[key]['pb'].update()  
        

    def reset_pbar(self, key):
        self.set_description_pbar(key, self.multi_pbar[key]['description'])
        self.multi_pbar[key]['pb'].reset()



    def update_iterable_pbar(self, key, new_iter, bar_fmt=None, description='', position=None):
        self.multi_pbar[key]['pb'].iterable = new_iter
        # position=position if not None else self.multi_pbar[key]['position'])
        # self.multi_pbar[key]['pb'] = tqdm.tqdm(new_iter, position=position if not None else self.multi_pbar[key]['position'])
                                                #  tqdm.tqdm( iterable=new_iter,
                                                # position=self.multi_pbar[key]['position'],
                                                # position=3,
                                                # bar_format=bar_fmt if bar_fmt is not None else self.bar_fmt) 
        # self.multi_pbar[key]['range'] = new_iter
        # self.set_description_pbar(key, description)


    def print_bar(self, key):
        for k, v in self.multi_pbar[key].items():
            tprint(f'{k}: {v}')
        


    def use_template(self, bar_len_dict, template='etv'):
        if template == 'etv':  # Epochs, Train, Validation
            epoch_key, train_key, test_key = 'epochs', 'Train', 'Validation'

            self.create_pbar(epoch_key, iter_tuple=bar_len_dict.get('e'), 
                                    bar_fmt='{desc}|{bar:24}| [{elapsed}<{remaining}]',
                                    description=f"Epoch: {bar_len_dict.get('e')[0]: >3}/{bar_len_dict.get('e')[1]: <3} AccL:-------% AccR:-------%")
            self.create_pbar(train_key, iter_tuple=bar_len_dict.get('t'), 
                                    description=f'{train_key}          AccL:-------% AccR:-------% ')
            self.create_pbar(test_key, iter_tuple=bar_len_dict.get('v'), 
                                    description=f'{test_key}     AccL:-------% AccR:-------% ')

        elif template == 'v':  # Epochs, Train, Validation
            test_key = 'Validation'
            self.create_pbar(test_key, iter_tuple=bar_len_dict.get('v'), 
                                    description=f'{test_key}     AccL:-------% AccR:-------% ')


        else:
            raise KeyError(f'Key {template} not found!')



def test_pbar():
    pb = PBar()

    pb.create_pbar('ep', iter_tuple=(0, 5), bar_fmt='Epoch: {n_fmt: >5}/{total_fmt} {desc}     |{bar:24}| [{elapsed}<{remaining}]', description='AccL:------% AccR:------%')
    # pb.set_description_pbar('ep', description='AccL:------% AccR:------%')

    for _ in pb.multi_pbar['ep']['pb']:
        time.sleep(1)
    



def test_multipbar():
     
    pb = PBar()

    pb.create_pbar('epochs', iter_tuple=(0,10), bar_fmt='Epoch: {n_fmt: >5} {desc} |{bar:24}| [{elapsed}<{remaining}]')
    pb.create_pbar('train', iter_tuple=(0,1000), description=f'number: {0: >5}')
    pb.create_pbar('test', iter_tuple=(0,1000), description=f'number: {0: >5}')


    tqdm.tqdm.write(str(pb.multi_pbar))

    for e in pb.multi_pbar['epochs']['pb']:
        for k in pb.multi_pbar.keys():
            if k=='epochs':
                continue
            for i in pb.multi_pbar[k]['range']:
                time.sleep(.001)
                pb.update_pbar(k, f'number: {i: >5} ')

        for k in pb.multi_pbar.keys():
            if k=='epochs':
                continue
            pb.reset_pbar(k)
        



    # tqdm.tqdm.write(str(pb.multi_pbar))
    tqdm.tqdm.write('DONE')



if __name__ == '__main__':
    # test_pbar()
    test_multipbar()


    pass
    


