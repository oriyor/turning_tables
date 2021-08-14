import re

def get_context_name(context):
    """
    :param context:
    :return: the name of the context in lower case
    """
    return context.context[0].title.replace(' ', '_')


def get_table(context):
    """
    :param context:
    :return: the context's table
    """
    for d in context.context:
        if len(d.table) > 0:
            return d
    return None


def get_images_from_context(context):
    """
    :param context:
    :return: the context's images
    """
    images = []
    for doc in context.context:
        if doc.metadata['type'] == 'image':
            images.append(doc)
        """
        if 'metadata' in doc['documents'] and 'type' in doc['documents']['metadata']:
            # image_docs = [doc for doc in context['context']['documents'] if doc['metadata']['type'] == 'image']
            images.append(doc['documents'])
        """
    if images:
        return images
    else:
        return None


def get_images_mapping_from_context(context):
    """
    :param context:
    :return: the coords of the images
    """
    images_map = {}
    for doc in context.context:
        if 'images_map' in doc.metadata:
            images_map = doc.metadata['images_map']
            break
        """
        if 'metadata' in doc['documents'] and 'type' in doc['documents']['metadata']:
            # image_docs = [doc for doc in context['context']['documents'] if doc['metadata']['type'] == 'image']
            images.append(doc['documents'])
        """
    if images_map:
        return images_map
    else:
        return None


REP_PATTERN = re.compile(r'\[[0-9]*\]|\n')
def normalize_paragraph_tag(p_tag):
    return re.sub(REP_PATTERN, "", p_tag.text)

def get_url_from_title(title):
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
