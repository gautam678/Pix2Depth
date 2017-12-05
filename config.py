CONFIG = {
        'development': False,
        'host': '128.143.63.199',
        'port': 8010,
        'pix2depth':{
                'first_option':'pix2pix',
                'second_option':'CycleGAN',
                'third_option':'CNN'
        },
        'depth2pix':{
                'first_option':'pix2pix',
                'second_option':'CycleGAN',
                #'third_option':'CNN'
        },
        'portrait':{
                'first_option': 'pix2pix',
                'second_option': 'CycleGAN',
                'third_option': 'CNN'
        }
}
