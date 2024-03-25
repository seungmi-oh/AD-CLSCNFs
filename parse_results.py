import pandas as pd
from config import get_args
from utils import makedirs, make_model_path
import os

def main(c):
    if c.dataset == 'mvtec':
        class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                    'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        run_type = [2,4,7,2,2,
                2,2,1,5,10,
                2,9,0,2,1] 
    if c.dataset == 'btad':
        class_names = ['01', '02', '03']
        run_type = [3,6,8]

    dir_feature=[]
    if c.test_data_type == 'aug':
        dir_feature.append(f'aug')
    if c.th_manual>0:
        dir_feature.append(f'manual_th_{c.th_manual:0.2f}')
    if c.is_open==True:
        dir_feature.append('imopen')
    if c.is_close==True:
        dir_feature.append('imclose')
    #
    df_all = None
    for class_idx, class_name in enumerate(class_names):
        c.class_name = class_name
        _ = make_model_path(c)
        model_dir = os.path.join(c.model_path, c.dataset, c.class_name, f'inp_{c.input_size}', c.model_name, f'run_{run_type[class_idx]}')
        save_dir_root = os.path.join(model_dir, c.class_name, c.infer_type)
        save_dir_root_dirs = os.listdir(save_dir_root)
        for save_dir_ in save_dir_root_dirs:
            if os.path.isdir(os.path.join(save_dir_root, save_dir_))==True:
                if 'joint' in c.infer_type:
                    if c.infer_type == save_dir_.split('-')[0]:
                        map_type = save_dir_
                    else:
                        pass
                elif c.infer_type=='fe_only':
                    if c.loss_type == save_dir_:
                        map_type = save_dir_
                    else:
                        pass
                else:
                    if c.infer_type == save_dir_:
                        map_type = save_dir_
                    else:
                        pass
        if len(dir_feature)>0:
            str_dir_features = '-'.join(dir_feature)
            tag = os.path.join(map_type, str_dir_features)
        else:
            tag = os.path.join(map_type, 'classic')
        
        csv_path_cl = os.path.join(model_dir, c.class_name, c.infer_type, tag, 'test_result.csv') 
        df_cl = pd.read_csv(csv_path_cl, index_col=0)
        if df_all is None:
            df_all = df_cl
        else:
            df_all = pd.concat([df_all, df_cl], axis=0)
    csv_path = os.path.join(c.model_path, c.dataset, 'results_total', f'inp_{c.input_size}-{c.model_name}', f'{c.infer_type}.csv') 
    makedirs(os.path.dirname(csv_path))
    df_all.to_csv(csv_path, header=True, float_format='%.2f')


if __name__ == '__main__':
    c = get_args()
    main(c)
