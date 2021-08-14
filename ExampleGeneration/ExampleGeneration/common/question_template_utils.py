

def substitute_named_templates(t, named_templates):
    for k, v in t.items():
        if isinstance(v, dict):
            if 'copy_from' in v:
                if v['copy_from'] in named_templates:
                    for k, tv in named_templates[v['copy_from']].items():
                        if k not in v:
                            v[k] = tv
                else:
                    assert (ValueError)
            else:
                substitute_named_templates(v, named_templates)

        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    if 'copy_from' in item:
                        if item['copy_from'] in named_templates:
                            for k, v in named_templates[item['copy_from']].items():
                                if k not in item:
                                    item[k] = v
                        else:
                            assert (ValueError)
                    else:
                        substitute_named_templates(item, named_templates)



def process_question_templates(templates):
    if templates[0]['name'] == "NamedTemplates":
        named_templates = templates[0]
        templates = templates[1:]

        for template in templates:
            substitute_named_templates(template, named_templates)

    return templates