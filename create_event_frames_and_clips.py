from nebfir.env import *
from nebfir.imop import (AETS, FRQ, SAE, SNN, TBR, FrameType, create_clips,
                                     create_event_frames)
from nebfir.imop.dataset_funcs import create_clips_nvfsd, create_event_frames_nvfsd

ev_rep = {
    'AETS':AETS,
    'FRQ':FRQ,
    'SAE':SAE,
    'SNN':SNN,
    'TBR':TBR
}

frame_type = {
    'events':FrameType.events,
    'grayscale':FrameType.grayscale
}


frame_rep_type = {
    'events':'AETS_40ms', ## PLACEHOLDER UNTIL BETTER WAY TO CALL
    'grayscale':'original'
}

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['SynFED', 'NVFSD'], help='Dataset')

parser.add_argument('--frames', '--create_frames', '--create_event_frames', action='store_true', dest='create_frames', help='Create event frames from event files')
parser.add_argument('--clips', '--create_clips',  action='store_true', dest='create_clips', help='Create frame clips from event frames')
parser.add_argument('-s', '-S', '--silent', '--SILENT',  action='store_false', dest='view_tqdm', help='Silent data creation')
parser.add_argument('-d', '-D', '--dry_run', '--DRY_RUN', '--dryrun', '--DRYRUN',  action='store_true', dest='dryrun', help='Dry run')
parser.add_argument('--summary', action='store_true', help='Prints options summary')

parser.add_argument('-p','--path','--PATH','--root_path', type=str, default='data/datasets/', dest='path', help='Data root path to events folder; i.e: "<PATH>/SynFED"')

parser.add_argument('-r','--events_representation', type=str, choices=['AETS', 'FRQ', 'SAE', 'SNN', 'TBR'], default='AETS', help='Events representation')
parser.add_argument('-t','--frame_type', type=str, choices=['events', 'grayscale'], default='events', help='Frame type')



# # # parser.add_argument('-V','-vvv', '--verbose', action="count", help="increase output verbosity")
# # # parser.add_argument('--foo', action=argparse.BooleanOptionalAction)
# # # parser.parse_args(['--no-foo'])


args = parser.parse_args()

# print(args.__dict__)
if args.summary:
    print(''.join([f'{k} : {v}\n' for k,v in args.__dict__.items()]))

if args.create_frames:
    if not args.frame_type == 'grayscale':
        if args.dataset == 'SynFED':
            for folder in ['s3dfm', 'deepfakes_v1']:
                create_event_frames(folder=folder, events_representation=ev_rep[args.events_representation], base_path=args.path, DRY_RUN=args.dryrun, view_tqdm=args.view_tqdm)
        elif args.dataset == 'NVFSD':
            create_event_frames_nvfsd(events_representation=ev_rep[args.events_representation], base_path=args.path, DRY_RUN=args.dryrun, view_tqdm=args.view_tqdm)
        else:
            raise ValueError(f'Dataset {args.dataset} does not exist !')
    else:
        print('WARNING: Grayscale frames cannot have event representation !\n')

if args.create_clips:
    if args.dataset == 'SynFED':
        for folder in ['s3dfm', 'deepfakes_v1']:
            create_clips(folder=folder, frame_type=frame_type[args.frame_type], frame_rep_type=frame_rep_type[args.frame_type],  base_path=args.path, DRY_RUN=args.dryrun, view_tqdm=args.view_tqdm)
    elif args.dataset == 'NVFSD':
        create_clips_nvfsd(frame_type=frame_type[args.frame_type], frame_rep_type=frame_rep_type[args.frame_type],  base_path=args.path, DRY_RUN=args.dryrun, view_tqdm=args.view_tqdm)
    else:
        raise ValueError(f'Dataset {args.dataset} does not exist !')
            
