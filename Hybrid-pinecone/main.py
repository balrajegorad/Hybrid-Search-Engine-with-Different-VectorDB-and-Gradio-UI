import gradio as gr
import pandas as pd
from scripts.db import get_mysql_conn
from scripts.query_embeddings import search_description

def search_by_name(name):
    conn = get_mysql_conn()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM products WHERE name LIKE %s", ("%" + name + "%",))
        results = cursor.fetchall()
    conn.close()
    if results:
        return pd.DataFrame(results, columns=["ID", "Name", "Price", "Category", "Description"])
    else:
        return "No products found."


def search_by_description(description):
    results = search_description(description, top_k=10)
    if results["matches"]:
        return pd.DataFrame(
            [
                {
                    "ID": match["id"],
                    "Name": match["metadata"]["name"],
                    "Description": match["metadata"]["description"],
                }
                for match in results["matches"]
            ]
        )
    else:
        return "No products found."


def search_by_price(min_price, max_price):
    conn = get_mysql_conn()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM products WHERE price BETWEEN %s AND %s", (min_price, max_price))
        results = cursor.fetchall()
    conn.close()
    if results:
        return pd.DataFrame(results, columns=["ID", "Name", "Price", "Category", "Description"])
    else:
        return "No products found."


with gr.Blocks() as demo:
    gr.Markdown("# Hybrid Product Search Engine")

    with gr.Tab("Search by Name"):
        name = gr.Textbox(label="Product Name")
        name_search_btn = gr.Button("Search")
        name_results = gr.Dataframe(label="Search Results")
        name_search_btn.click(search_by_name, inputs=name, outputs=name_results)

    with gr.Tab("Search by Description"):
        description = gr.Textbox(label="Product Description")
        desc_search_btn = gr.Button("Search")
        desc_results = gr.Dataframe(label="Search Results")
        desc_search_btn.click(search_by_description, inputs=description, outputs=desc_results)

    with gr.Tab("Search by Price Range"):
        min_price = gr.Number(label="Minimum Price")
        max_price = gr.Number(label="Maximum Price")
        price_search_btn = gr.Button("Search")
        price_results = gr.Dataframe(label="Search Results")
        price_search_btn.click(search_by_price, inputs=[min_price, max_price], outputs=price_results)

demo.launch()
