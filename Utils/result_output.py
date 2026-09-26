def output_dict(result_dict, title='', func=print, metrics=None, demical=4, form='f', space='  '):
    output_str = title
    if metrics is None:
        output_list = [(k, v) for k, v in result_dict.items()]
    else:
        output_list = [(k, result_dict[k]) for k in metrics if k in result_dict]
    for k, v in output_list:
        if isinstance(v, list):
            res = sum(v) / len(v)
        else:
            res = v
        output_str += f"{k}={res:.{demical}{form}}{space}"
    func(output_str)
