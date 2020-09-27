# -*- coding: utf-8 -*-
from app import app
from layout import main_layout

app.layout = main_layout

import callbacks

if __name__ == '__main__':
    app.run_server(debug=True)
