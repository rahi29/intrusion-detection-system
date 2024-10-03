from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import Logistic_Regression
import csv
import numpy as np
import pandas as pd

def filters(data):
    

    columns_to_remove = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'is_guest_login','su_attempted','wrong_fragment','urgent','hot','num_failed_logins','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','srv_diff_host_rate']

    data = data.drop(columns=columns_to_remove)

    data = data.loc[:, (data != 0).any(axis=0)]

    data.to_csv('./Data/filtered_data.csv', index=False)

    return data

# Define the HTTP request handler class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    # GET method handler
    def do_GET(self):
        # Parse the URL
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        # Check the path and respond accordingly
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_form_page().encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    # Function to generate HTML page with form
    def generate_form_page(self):
        form_html = """
        <html>
        <head>
            <style>
                body {
                    text-align: center; /* Center align the content */
                }
                form {
                    margin-top: 50px; /* Add margin to the top of the form */
                }
                input[type="file"] {
                    margin-bottom: 20px; /* Add margin between the file input and the submit button */
                }
            </style>
        </head>
        <body>
            <h1>Upload CSV File</h1>
            <form action="/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv"><br>
                <input type="submit" value="Upload">
            </form>
            <br>
            %s
        </body>
        </html>
        """ % self.generate_table_from_csv()
        return form_html

    # POST method handler for file upload
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Extract the boundary from Content-Type header
        content_type = self.headers['Content-Type']
        boundary = content_type.split('; ')[1].split('=')[1].encode()

        # Split the post data using the boundary
        sections = post_data.split(b"--" + boundary)
        # sections= sections.split(b'text/csv')
        # sections.remove(b'Content-Disposition: form-data; name="file"; filename="Test_data.csv" Content-Type: text/csv')

        start_pos = post_data.find(boundary)

        # Find the start position of the CSV data
        csv_start_pos = post_data.find(b'\r\n\r\n', start_pos) + len(b'\r\n\r\n')

        # Extract CSV data
        csv_data = post_data[csv_start_pos:]
        
        # Save the uploaded file as data.csv
        with open('data.csv', 'w') as f:
            for x in sections:
                f.write(x.decode())
        print('pass1')
        with open('data.csv','r') as r:
            x=r.readlines()
            x=x[4:-2]
            print('pass2')
            with open('data.csv','w') as w:
                for y in x:
                    w.write(y)
        print('pass3')

        
        data = pd.read_csv('data.csv')
        data=filters(data)
        y_test=Logistic_Regression.train(data)
    
        # Sample 1D array
        

        # Define a function to convert 0s and 1s to 'normal' and 'unnormal'
        def convert_to_status(value):
            return 'normal' if value == 1 else 'anomaly'

        # Apply the function element-wise to the array
        converted_data = np.vectorize(convert_to_status)(y_test)

        # Load CSV data into a pandas DataFrame
        csv_data = pd.read_csv('data.csv')

        # Create a DataFrame from the converted array
        converted_df = pd.DataFrame(converted_data, columns=['class'])

        # Merge the two DataFrames based on their indices
        merged_data = pd.concat([csv_data, converted_df], axis=1)
        merged_data.to_csv('data.csv', index=False)

        self.send_response(303)
        self.send_header('Location', '/')
        self.end_headers()

    # Function to generate HTML table from CSV
    def generate_table_from_csv(self):
        try:
            with open('data.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                table_html = "<h2 class>Uploaded CSV Table</h2><table border='1'>"
                for row in reader:
                    table_html += "<tr>"
                    for col in row:
                        table_html += f"<td>{col}</td>"
                    table_html += "</tr>"
                table_html += "</table>"
            return table_html
        except FileNotFoundError:
            return ""

# Define the server address and port
host = 'localhost'
port = 8000

# Create an HTTP server and assign the request handler
server = HTTPServer((host, port), SimpleHTTPRequestHandler)
print(f"Server running on {host}:{port}")

# Start the server
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.shutdown()
    print("\nServer stopped.")