#!/usr/bin/env python3
"""
Launches the neural-style application server.

@author: Álvaro Barbero Jiménez
"""

import connexion

if __name__ == '__main__':
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.add_api('swagger.yaml', arguments={'title': 'Blends content and style images in an artistic way, using Deep Neural Networks'})
    app.run(port=80, debug=True)
