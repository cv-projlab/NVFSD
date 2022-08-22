from data.inp.enums import TaskRecordingCombinations as TRC
from nebfir.config.configurations import Configurations
from nebfir.env import *
from nebfir.tools.tools_basic import (multi_split,
                                      separators_letters_punctuation,
                                      str2int_list)
from nebfir.tools.tools_path import create_mid_dirs


class FrameType(Enum):
    events_aets40 = 'AETS_40ms'
    grayscale = 'original'

frame_type = {
    'events_aets40':FrameType.events_aets40,
    'grayscale':FrameType.grayscale
}

def create_list_with_all_data(frametype=FrameType.events_aets40, path = 'data/datasets/', *args, **kwargs):
    print('CREATING BASE LIST ...')
    
    #############################################################################
    #############################################################################
    #############################################################################
    
    framesPerClip=12
    stride = 1 if frametype is FrameType.events_aets40 else 40 # 40 for grayscale to match the framerate of event frames. int((24-12)/1-1) == int((500-12)/40-1)
    folders={'s3dfm':1,'deepfakes_v1':0}
    

    #############################################################################
    #############################################################################
    #############################################################################

    filename=f'data/inp/lists/SynFED_df_all_{frametype.name}.csv'


    lines=[]
    for folder, folder_value in folders.items():
        paths = sorted(glob(os.path.join(path, f'SynFED/Clips/{folder}/{frametype.value}/*.*'), recursive=True), key=lambda x: multi_split(x, string.ascii_letters+string.punctuation, str2int_list))

        user_rec_list = list(map(lambda x: multi_split( x, separators=string.ascii_letters+string.punctuation, map_func=str2int_list)[~1:], paths))

        for i, (path_, (user, recording)) in enumerate(zip(paths, user_rec_list)):
            real = 1 if ((user == recording) and folder_value == 0) or folder_value == 1 else 0  # fakes of own == real -> same dynamic
            lines.append([ folder, path_, user, recording, real])
            
    paths_df = pd.DataFrame(lines, columns=['folder', 'path', 'user', 'rec', 'real'])#.sort_values(by=['user', 'rec'], ascending=True)
    indexes = paths_df.index

    create_mid_dirs(filename)
    with open(filename, 'w') as f:
        f.write('folder;path;frameno_list;user;rec;real\n')

    lines = []
    for i, idx in enumerate(tqdm(indexes)):
        line = paths_df.iloc[idx]
        
        folder = line['folder']
        path = line['path']
        user = line["user"]
        rec = line["rec"]
        real = line["real"]
        
        clip = np.load(path) # shape: (channels, noframes, H, W)
        
        noFrames = clip.shape[1]
        noClips = int(np.floor((noFrames - framesPerClip)/stride + 1))
        
        for k in range(noClips):
            frames_list = list(np.arange(stride*k , stride*k + framesPerClip)) 

            write_line = ';'.join([folder ,path ,str(frames_list) ,str(user) ,str(rec) ,str(real)]) + '\n'
            lines.append(write_line)
            
        if (i % 500 == 0) or (i == len(indexes)-1):
            with open(filename, 'a') as f:
                f.writelines(lines)
            lines = []

def create_list(frametype = FrameType.events_aets40, userno=30, recno=10, fakeno=15, train_split_frac=.7, authentics=True, impostors=True, generate_configuration_file=False, *args, **kwargs):
    print('CREATING TRAIN AND TEST LIST ...')
    
    rand_state = 0
    np.random.seed(rand_state)

    def append_samples2df(samples, df):
        return pd.concat([df, samples], ignore_index=True)
        
    def get_vals(df):
        folders = '-'.join(sorted(df['folder'].unique()))
        users = f"{df['user'].min()}--{df['user'].max()}"
        recordings = f"{df['rec'].min()}--{df['rec'].max()}"

        return folders, users, recordings

    #################################################################################################################
    #################################################################################################################
    #################################################################################################################

    assert 1< userno <76, f'Lists must contain at least 1 user and at most 76 users. Got userno={userno}'
    assert fakeno < userno, f'Fake users number [fakeno] ({fakeno}) must be inferior to Real users number [userno] ({userno})'
    assert 1<= recno <=10, f'Real recordings [recno] must be in range (1,10). Got recno={recno}'
    assert 0. < train_split_frac <1., f'Split fraction must be in range (0., 1.). Got train_split_frac={train_split_frac}'
    assert authentics or impostors, f'Lists must contain authentics, impostors or both. Got authentics={authentics}, impostors={impostors}'

    stride = 1 if frametype is FrameType.events_aets40 else 40

    #################################################################################################################
    #################################################################################################################
    #################################################################################################################


    print('frametype: ', frametype)
    print()
    print('userno: ',userno)
    print('recno: ',recno)
    print('fakeno: ',fakeno)
    print('train_split_frac: ', train_split_frac)
    print()





    real_users_train = np.ones((userno,recno), dtype=int) * np.arange(recno)
    fake_users_train = np.array( [np.random.choice(list(set(range(userno))-set([u])) , fakeno, replace=False) for u in range(userno)] ) # Random choice fake user * fakeno  for each real user for train

    assert all([len(set(users)) == fakeno for users in fake_users_train]), f'Users are not unique! {[f"fake users number:{len(set(users))}; fakeno:{fakeno}" for users in fake_users_train]}'



    print('user', '| user recordings', '| fake users')
    for i in range(userno):
        print(i, real_users_train[i], sorted(fake_users_train[i]))
        # break
    print()


    df_all = pd.read_csv(f'data/inp/lists/SynFED_df_all_{frametype.name}.csv', sep=';')
    print('df_all shape: ', df_all.shape)
    print('df_all folders, users, recordings', get_vals(df_all))
    print()


    ## TRAIN TEST SPLIT
    df_all_train = df_all.groupby(['path','rec','user']).sample(frac=train_split_frac, replace=False, random_state=rand_state) ## Split normalized 

    df_all_train.sort_values(by=['user', 'rec', 'path', 'frameno_list'], ascending=True, inplace=True)
    df_all_test = df_all.drop(df_all_train.index)
    df_all_test.sort_values(by=['user', 'rec', 'path', 'frameno_list'], ascending=True, inplace=True)


    print('df_all_train shape: ', df_all_train.shape)
    print('df_all_test shape: ', df_all_test.shape)
    print()


    df_train = pd.DataFrame(columns=df_all.columns)
    df_test = pd.DataFrame(columns=df_all.columns)


    for dynamic in range(userno):
        # TRAIN
        if authentics:
            df_train = append_samples2df(df_all_train[ (df_all_train['folder']=='s3dfm') & (df_all_train['user']==dynamic) & (df_all_train['rec'].isin(real_users_train[dynamic])) ], df_train)
        
        if impostors:
            df_train = append_samples2df(df_all_train[ (df_all_train['folder']=='deepfakes_v1') & (df_all_train['rec']==dynamic) & (df_all_train['user'].isin(fake_users_train[dynamic])) ], df_train)
        
    ## TEST
    if authentics:
        df_test = append_samples2df(df_all_test[ (df_all_test['folder']=='s3dfm') & (df_all_test['user'].isin( list(range(userno))) ) & (df_all_test['rec'].isin( list(range(userno)) ) )] , df_test) # ALL REAL
    if impostors:
        df_test = append_samples2df(df_all_test[ (df_all_test['folder']=='deepfakes_v1') & (df_all_test['user'].isin( list(range(userno))) ) & (df_all_test['rec'].isin( list(range(userno)) ) )] , df_test) # ALL FAKES


    print('df_train shape: ', df_train.shape)
    print('df_test shape: ', df_test.shape)
    print()

    for name, df__ in {'df_train':df_train, 'df_test':df_test}.items():
        print(name, 'stats:')
        print(name, ' folders, users, recordings', get_vals(df__))

        df__.sort_values(by=['user', 'rec', 'folder' ])

        print('unique folder shape:' , df__['folder'].unique().shape)
        print('unique user shape:' , df__['user'].unique().shape)
        print('unique rec shape:' , df__['rec'].unique().shape)
        print('unique frameno_list shape:' , df__['frameno_list'].unique().shape)
        print('unique path shape:' , df__['path'].unique().shape)
        print()


    # SAVE TRAIN
    split = 'TRAIN'
    folders, users, recordings = get_vals(df_train)
    train_save_name=f'data/inp/lists/SynFED_{split}_{frametype.name}_test-split{(1-train_split_frac)*100:0.0f}_S{stride}_F{folders}_U{users}_R{recordings}_realno-{recno}_fakeno-{fakeno}.csv'
    print(os.path.join(os.getcwd(), train_save_name))
    df_train.to_csv(train_save_name, sep=',', index=False); print('Saved !')


    # SAVE TEST
    split = 'TEST'
    folders, users, recordings = get_vals(df_test)
    test_save_name=f'data/inp/lists/SynFED_{split}_{frametype.name}_test-split{(1-train_split_frac)*100:0.0f}_S{stride}_F{folders}_U{users}_R{recordings}.csv'
    print(os.path.join(os.getcwd(), test_save_name))
    df_test.to_csv(test_save_name, sep=',', index=False); print('Saved !')

    if generate_configuration_file:
        generate_config(userno, tlist=os.path.join(os.getcwd(), train_save_name), ttlist=os.path.join(os.getcwd(), test_save_name), frametype=frametype, dataset_name='SynFED')

def generate_config(userno, tlist, ttlist, frametype, dataset_name):
    config={
        'trainer':{ 'model':{ 'num_classes': userno }},
        'data':{
                'dataset':{
                            'dataset_name': dataset_name,
                            'dT': 40,
                            'subclipT': 500,
                            'stride': 1,
                            'type': frametype.name,
                            },
                'lists':{
                        'train': tlist,
                        'test': ttlist,
                        }
            },
    }

    
    existing_configs = list(map(lambda x: multi_split(x, string.punctuation+string.ascii_letters, map_func=str2int_list), glob('configs/config-*.yml')))
    if not existing_configs:
        name = f'config-0.yml'
    else:
        val = max( existing_configs )[0] + 1
        name = f'config-{val}.yml'
    
    Configurations.save_config(config=config, filename=name)
    
    print(f'Automatically generated {name} !')

    

def create_list_with_all_data_nvfsd(frametype=FrameType.events_aets40, path = 'data/datasets/', *args, **kwargs):
    print('CREATING BASE LIST ...')
    
    #############################################################################
    #############################################################################
    #############################################################################
    
    framesPerClip=12
    stride = 1 if frametype is FrameType.events_aets40 else 40 # 40 for grayscale to match the framerate of event frames. int((24-12)/1-1) == int((500-12)/40-1)
    

    #############################################################################
    #############################################################################
    #############################################################################

    filename=f'data/inp/lists/NVFSD_df_all_{frametype.name}.csv'

    paths = sorted(glob(os.path.join(path, f'NVFSD/Clips/AETS_40ms/*.*'), recursive=True), key=lambda x: multi_split(x, separators_letters_punctuation, str2int_list))
    
    user_task_rec_list = list(map(lambda x: multi_split( x, separators=separators_letters_punctuation, map_func=str2int_list)[~2:], paths))
    

    lines=[]
    for i, (path, (user, task, recording)) in enumerate(zip(paths, user_task_rec_list)):
        real = 1
        lines.append([ path, user, task, recording, real])
        
    


    paths_df = pd.DataFrame(lines, columns=['path', 'user', 'task', 'rec', 'real'])#.sort_values(by=['user', 'rec'], ascending=True)
    indexes = paths_df.index

    create_mid_dirs(filename)
    with open(filename, 'w') as f:
        f.write('path;frameno_list;user;task;rec;real\n')

    lines = []
    for i, idx in enumerate(tqdm(indexes)):
        line = paths_df.iloc[idx]
        
        
        path = line['path']
        user = line["user"]
        task = line["task"]
        rec = line["rec"]
        real = line["real"]
        
        clip = np.load(path) # shape: (channels, noframes, H, W)
        
        noFrames = clip.shape[1]
        noClips = int(np.floor((noFrames - framesPerClip)/stride + 1))
        
        for k in range(noClips):
            frames_list = list(np.arange(stride*k , stride*k + framesPerClip)) 

            write_line = ';'.join([path ,str(frames_list) ,str(user) ,str(task) ,str(rec) ,str(real)]) + '\n'
            lines.append(write_line)
            
        if (i % 500 == 0) or (i == len(indexes)-1):
            with open(filename, 'a') as f:
                f.writelines(lines)
            lines = []

def create_list_nvfsd(frametype = FrameType.events_aets40, userno=30, combination: str = 'A1B1C123', train_split_frac=.7, generate_configuration_file=False, *args, **kwargs):
    print('CREATING TRAIN AND TEST LIST ...')
    
    rand_state = 0
    np.random.seed(rand_state)

    def append_samples2df(samples, df):
        return pd.concat([df, samples], ignore_index=True)
        
    def get_vals(df):
        users = f"{df['user'].min()}--{df['user'].max()}"
        tasks = f"{df['task'].min()}--{df['task'].max()}"
        recordings = f"{df['rec'].min()}--{df['rec'].max()}"

        return users, tasks, recordings

    #################################################################################################################
    #################################################################################################################
    #################################################################################################################

    TRC.assert_combination_exists(combination)
    combination = TRC[combination]
    assert 1< userno <=40, f'Lists must contain at least 1 user and at most 40 users. Got userno={userno}'
    assert 0. < train_split_frac <1., f'Split fraction must be in range (0., 1.). Got train_split_frac={train_split_frac}'

    stride = 1 if frametype is FrameType.events_aets40 else 40

    #################################################################################################################
    #################################################################################################################
    #################################################################################################################


    print('frametype: ', frametype)
    print()
    print('combination: ',combination)
    print('train_split_frac: ', train_split_frac)
    print()




    df_all = pd.read_csv(f'data/inp/lists/NVFSD_df_all_{frametype.name}.csv', sep=';')
    print('df_all shape: ', df_all.shape)
    print('df_all users, tasks, recordings', get_vals(df_all))
    print()

    df_all_aux = df_all
    ## TRAIN TEST SPLIT
    df_all_train = df_all.groupby(['path','rec','task', 'user']).sample(frac=train_split_frac, replace=False, random_state=rand_state) ## Split normalized 

    df_all_train.sort_values(by=['user', 'task', 'rec', 'path', 'frameno_list'], ascending=True, inplace=True)
    df_all_test = df_all.drop(df_all_train.index)
    df_all_test.sort_values(by=['user', 'task', 'rec', 'path', 'frameno_list'], ascending=True, inplace=True)


    print('df_all_train shape: ', df_all_train.shape)
    print('df_all_test shape: ', df_all_test.shape)
    print()



    mask_combination=[]
    for i, task in enumerate(combination.value['t']):       
        for recording in combination.value['r'][i]:
            mask_combination.append( [task, recording] )

    df_mask = " | ".join([f"(df_all['user'].isin(list(range({userno})))) & (df_all['task'] == {t}) & (df_all['rec'] == {r})" for t, r in mask_combination])



    df_train = df_all_train[ eval(df_mask) ]
    df_test = df_all_test[ eval(df_mask) ]

    df_full = df_all_aux[ eval(df_mask) ]

        
    

    print('df_train shape: ', df_train.shape)
    print('df_test shape: ', df_test.shape)
    print('df_full shape: ', df_full.shape)
    print()

    for name, df__ in {'df_train':df_train, 'df_test':df_test}.items():
        print(name, 'stats:')
        print(name, ' users, tasks, recordings', get_vals(df__))

        df__.sort_values(by=['user', 'task', 'rec' ])

        print('unique user shape:' , df__['user'].unique().shape)
        print('unique task shape:' , df__['task'].unique().shape)
        print('unique rec shape:' , df__['rec'].unique().shape)
        print('unique frameno_list shape:' , df__['frameno_list'].unique().shape)
        print('unique path shape:' , df__['path'].unique().shape)
        print()


    # SAVE TRAIN
    split = 'TRAIN'
    users, tasks, recordings = get_vals(df_train)
    print(split, 'users:', users, 'tasks:', tasks, 'recordings:', recordings)
    train_save_name=f'data/inp/lists/NVFSD_{split}_{frametype.name}_test-split{(1-train_split_frac)*100:0.0f}_S{stride}_U{users}_combination{combination.name}.csv'
    print(os.path.join(os.getcwd(), train_save_name))
    df_train.to_csv(train_save_name, sep=',', index=False); print('Saved !')


    # SAVE TEST
    split = 'TEST'
    users, tasks, recordings = get_vals(df_test)
    print(split, 'users:', users, 'tasks:', tasks, 'recordings:', recordings)
    test_save_name=f'data/inp/lists/NVFSD_{split}_{frametype.name}_test-split{(1-train_split_frac)*100:0.0f}_S{stride}_U{users}_combination{combination.name}.csv'
    print(os.path.join(os.getcwd(), test_save_name))
    df_test.to_csv(test_save_name, sep=',', index=False); print('Saved !')

    # SAVE FULL
    split = 'FULL'
    users, tasks, recordings = get_vals(df_full)
    print(split, 'users:', users, 'tasks:', tasks, 'recordings:', recordings)
    full_save_name=f'data/inp/lists/NVFSD_{split}_{frametype.name}_S{stride}_U{users}_combination{combination.name}.csv'
    print(os.path.join(os.getcwd(), full_save_name))
    df_full.to_csv(full_save_name, sep=',', index=False); print('Saved !')

    

    if generate_configuration_file:
        generate_config(userno, tlist=os.path.join(os.getcwd(), train_save_name), ttlist=os.path.join(os.getcwd(), test_save_name), frametype=frametype, dataset_name='NVFSD')

    
            
if __name__ == '__main__':
    funcs = {   'SynFED': {
                        'create_base_list':create_list_with_all_data,
                        'create_list':create_list }, 
                'NVFSD': {
                        'create_base_list':create_list_with_all_data_nvfsd,
                        'create_list':create_list_nvfsd }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['SynFED', 'NVFSD'], help='Dataset')

    parser.add_argument('function', type=str, nargs='+', choices={'create_base_list', 'create_list'}, help='Function to run')
    parser.add_argument('-t','--frame_type','--frametype', '--type', type=str, choices=['events_aets40', 'grayscale'], dest='frame_type', default='events_aets40', help='Frame type')

    parser.add_argument('-p','--path','--PATH','--root_path', type=str, default='data/datasets/', dest='path', help='Data root path to dataset folder; i.e: "<PATH>/SynFED"')

    parser.add_argument('--userno', type=int, default=30, help='Number of users')
    parser.add_argument('--train_split_frac', type=float, default=.7, help='Train split fraction')

    parser.add_argument('-c', '--comb', '--combination', type=str, default='COMBINATION', dest='combination', help='NVFSD: Task Recording Combination')

    parser.add_argument('--recno', type=int, default=10, help='SynFED: Real recordings number')
    parser.add_argument('--fakeno', type=int, default=15, help='SynFED: Fake users number')
    parser.add_argument('--no-real', '--no-authentic', action='store_true', help='SynFED: Creates train and test list without real dynamics')
    parser.add_argument('--no-fake', '--no-impostor', action='store_true', help='SynFED: Creates train and test list without fake dynamics')
    
    parser.add_argument('--summary', action='store_true', help='Prints options summary')
    parser.add_argument('--gen_cfg','--gen_conf','--gen_config','--generate_configuration_file', action='store_true', dest='generate_configuration_file', help='Automaticaly generate config file')
    args = parser.parse_args()

    if len(args.function) > 2: raise ValueError(f'Invalid number of functions! Expected function number <=2. Got {len(args.function)}, {args.function}')
    if len(set(args.function)) != len(args.function): raise ValueError(f'Invalid functions! Function names must be different! Got {args.function}')

    if args.summary:
        print(''.join([f'{k} : {v}\n' for k,v in args.__dict__.items()]))


    for func_name in args.function:
        funcs[args.dataset][func_name]( frametype=frame_type[args.frame_type], 
                                        path=args.path,

                                        userno=args.userno, 
                                        train_split_frac=args.train_split_frac, 

                                        recno=args.recno, 
                                        fakeno=args.fakeno, 
                                        authentics=not args.no_real, 
                                        impostors=not args.no_fake,
                                        
                                        combination=args.combination,
                                        
                                        generate_configuration_file=args.generate_configuration_file,


                            )


