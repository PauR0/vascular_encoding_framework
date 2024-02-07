#! /usr/bin/env python3



def split_and_pad(input_string, max_length=80, pad_direction='right', pad_symb='.'):

    if not input_string:
        return []

    if len(input_string) < max_length:
        if pad_direction == 'left':
            return [input_string.rjust(max_length, pad_symb)]
        elif pad_direction == 'right':
            return [input_string.ljust(max_length, pad_symb)]
        else:
            return [input_string]

    last_space = input_string.rfind(' ', 0, max_length + 1)

    if last_space != -1:
        return [input_string[:last_space]] + split_and_pad(input_string[last_space + 1:], max_length, pad_direction)
    else:
        last_chunk = input_string[:max_length]
        if pad_direction == 'right':
            last_chunk = last_chunk.ljust(max_length, pad_symb)
        elif pad_direction == 'left':
            last_chunk = last_chunk.rjust(max_length, pad_symb)
        return [last_chunk] + split_and_pad(input_string[max_length:], max_length, pad_direction)
#

def computing_message(task='', prnt=True):
    comp_msg = f"Computing {task} "
    strout = '\n'.join(split_and_pad(comp_msg, pad_direction='right'))
    if prnt:
        print(strout)
    return strout
#

def done_message(task='', prnt=True):
    done_msg = f" {task} done!"
    strout = '\n'.join(split_and_pad(done_msg, pad_direction='left'))
    if prnt:
        print(strout)
    return strout
#

def error_message(info, prnt=True):
    err_msg = f"ERROR: {info}"
    strout = '\n'.join(split_and_pad(err_msg, pad_direction='right'))
    if prnt:
        print(strout)
    return strout
#
