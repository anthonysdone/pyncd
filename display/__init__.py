import display.node_category as dnc
import data_structure.Category as cat

def print_category(target: cat.BroadcastedCategory) -> None:
    text = dnc.display_category(target).render()
    print(text)