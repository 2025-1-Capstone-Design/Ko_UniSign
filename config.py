mt5_path = "google/mt5-base" #"./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "KO_WORD_SYN": "./data/KO_WORD_SYN/labels-3000.train",
                    "KO_SEN_SYN": "./data/KO_SEN_SYN/labels.train",
                    "KO_SEN_REAL": "./data/KO_SEN_REAL/labels.train"
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "KO_WORD_SYN": "./data/KO_WORD_SYN/labels-3000.dev",
                    "KO_SEN_SYN": "./data/KO_SEN_SYN/labels.dev",
                    "KO_SEN_REAL": "./data/KO_SEN_REAL/labels.dev"
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "KO_WORD_SYN": "./data/KO_WORD_SYN/labels-3000.test",
                    "KO_SEN_SYN": "./data/KO_SEN_SYN/labels.test",
                    "KO_SEN_REAL": "./data/KO_SEN_REAL/labels.test"
                    }


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "KO_WORD_SYN": "./dataset/KO_WORD_SYN/rgb_format",
            "KO_SEN_SYN": "./dataset/KO_SEN_SYN/rgb_format",
            "KO_SEN_REAL": "./dataset/KO_SEN_REAL/rgb_format"
            }

# pose paths
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "KO_WORD_SYN": "./dataset/KO_WORD_SYN/pose_format",
            "KO_SEN_SYN": "./dataset/KO_SEN_SYN/pose_format",
            "KO_SEN_REAL": "./dataset/KO_SEN_REAL/pose_format"
            }