mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    'OpenASL': "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_label/labels.train",
                    'OpenASL_QA': "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_label_QA/labels.train"
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    'OpenASL': "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_label/labels.dev",
                    'OpenASL_QA': "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_label_QA/labels.dev"
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    'OpenASL': "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_label/labels.test",
                    'OpenASL_QA': "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_label_QA/labels.test"
                    }


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "OpenASL": "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl/video-clip",       
            "OpenASL_QA": "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl/video-clip"            
            }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "OpenASL": "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_keypoints",
            "OpenASL_QA": "/home/dial/jonghyo/SignQA/Data/OpenASL/openasl_keypoints"
            }