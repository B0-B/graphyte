from core.graph import GraphLike, BaseTree
from IPython.display import Markdown, display, HTML

def show (graph: GraphLike) -> None:

    '''
    Renders the graph in a jupyter notebook cell.
    '''

    # convert to mermaid code
    direction = 'TB' if isinstance(graph, BaseTree) else 'LR'
    markdown_payload = f"# {graph.name}\n\n" + graph.to_mermaid(direction=direction)

    # Render Markdown
    rendered_md = Markdown(markdown_payload)

    # diplay render
    display(rendered_md)