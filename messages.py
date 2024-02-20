#! /usr/bin/env python3

import os

LENGTH = os.get_terminal_size()[0]

def split_and_pad(input_string, max_length=LENGTH, pad_direction='right', pad_symb='.'):

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

def get_message_maker(main_message, pad_direction='right'):

    def message_maker(info='', prnt=True):
        if pad_direction is None:
            strout = f"{main_message} {info}"
            if prnt:
                print(strout)
            return strout
        if pad_direction == 'right':
            message = f"{main_message} {info}"
        else:
            message = f" {info} {main_message}"
        strout = '\n'.join(split_and_pad(message, pad_direction=pad_direction))
        if prnt:
            print(strout)
        return strout
    #
    return message_maker
#

computing_message = get_message_maker(main_message="Computing")
done_message      = get_message_maker(main_message='done!', pad_direction='left')
error_message     = get_message_maker(main_message="ERROR: ")
warning_message   = get_message_maker("WARNING: ")
info_message      = get_message_maker(main_message="INFO: ", pad_direction=None)
