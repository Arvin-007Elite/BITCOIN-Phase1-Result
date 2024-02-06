from flask import Flask, render_template

import nbformat
from nbconvert import HTMLExporter

app = Flask(__name__, template_folder='templates')

# Convert Jupyter Notebook to HTML
def notebook_to_html(notebook_path):
    with open(notebook_path, 'r') as nb_file:
        nb_content = nb_file.read()
    nb = nbformat.reads(nb_content, as_version=4)
    html_exporter = HTMLExporter()
    html_content, _ = html_exporter.from_notebook_node(nb)
    return html_content

@app.route('/')
def jupyter_in_flask():
    notebook_path = 'Sentiment-Bitcoin.ipynb'
    jupyter_html = notebook_to_html(notebook_path)
    return render_template('jupiter_template.html', jupyter_html=jupyter_html)

if __name__ == '__main__':
    app.run(debug=True)








# @app.route("/")
# @app.route("/index")
# # @app.route("/index")
# def index():
#     return render_template("index.html")
#     # return "<h1>Hello World</h1"


# @app.route("/about")
# def about():
#     return "<h1>About</h1>"