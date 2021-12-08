from model.shitf_generator import ShiftGenerator

def build_shift_generator(cfg, input_shape):
    return ShiftGenerator(cfg, input_shape)