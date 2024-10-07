import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import base64

# Directories containing saved outputs
image_dir = 'generated_images/individual'
point_cloud_dir = 'point_clouds'
persistence_diagram_dir = 'persistence_diagrams'

# Load losses data (from CSV)
loss_data = pd.read_csv('losses.csv')

# Load intrinsic dimensions data (from CSV)
intrinsic_dim_data = pd.read_csv('intrinsic_dimensions.csv')

# Get list of epochs based on saved images (every 10 epochs)
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith('.png') and 'epoch_' in f],
    key=lambda x: int(x.split('_')[1])
)
epochs = sorted(list(set(int(f.split('_')[1]) for f in image_files if 'epoch_' in f)))

max_epoch = max(epochs)
min_epoch = min(epochs)

# Create a mapping from epoch to file paths
epoch_to_image = {epoch: [f for f in image_files if f'epoch_{epoch:03d}' in f] for epoch in epochs}
epoch_to_point_cloud_csv = {
    epoch: os.path.join(point_cloud_dir, f'point_cloud_epoch_{epoch:03d}.csv') for epoch in epochs
}
epoch_to_persistence_csv = {
    epoch: os.path.join(persistence_diagram_dir, f'persistence_epoch_{epoch:03d}.csv') for epoch in epochs
}

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "DCGAN Training Progress with Interactive TDA"

# App Layout
app.layout = html.Div([
    html.H1("DCGAN Training Progress on CIFAR-10 Class 0 with Interactive TDA"),
    
    # Slider for selecting epoch (increments of 10)
    dcc.Slider(
        id='epoch-slider',
        min=min_epoch,
        max=max_epoch,
        step=25,
        value=min_epoch,
        marks={epoch: str(epoch) for epoch in epochs}
    ),
    
    # Display current epoch
    html.Div(id='current-epoch', style={'textAlign': 'center', 'fontSize': 24, 'margin': '20px'}),
    
    # Graphs for Point Cloud and Persistence Diagram
    html.Div([
        html.Div([
            html.H3("Point Cloud"),
            dcc.Graph(id='point-cloud-graph', config={'displayModeBar': False}, style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        
        html.Div([
            html.H3("Persistence Diagram"),
            dcc.Graph(id='persistence-diagram-graph', config={'displayModeBar': False}, style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),
    
    # Display Corresponding Images
    html.Div([
        html.H3("Corresponding Images"),
        html.Div(id='corresponding-images', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
    # Loss graph
    html.Div([
        html.H3("Training Losses"),
        dcc.Graph(id='loss-graph', config={'displayModeBar': False})
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
    # Intrinsic dimensions graph
    html.Div([
        html.H3("Intrinsic Dimensions"),
        dcc.Graph(id='intrinsic-dim-graph', config={'displayModeBar': False})
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    
], style={'width': '90%', 'margin': 'auto', 'textAlign': 'center'})

# Callback to update graphs based on slider and selections
@app.callback(
    [
        Output('point-cloud-graph', 'figure'),
        Output('persistence-diagram-graph', 'figure'),
        Output('corresponding-images', 'children'),
        Output('current-epoch', 'children'),
        Output('loss-graph', 'figure'),
        Output('intrinsic-dim-graph', 'figure')
    ],
    [
        Input('epoch-slider', 'value'),
        Input('point-cloud-graph', 'selectedData'),
        Input('persistence-diagram-graph', 'selectedData')
    ]
)
def update_content(selected_epoch, selected_point_cloud_data, selected_persistence_data):
    # Update current epoch text
    epoch_text = f'Epoch: {selected_epoch}'

    # Load point cloud data
    point_cloud_csv = epoch_to_point_cloud_csv.get(selected_epoch, '')
    if os.path.exists(point_cloud_csv):
        point_cloud_df = pd.read_csv(point_cloud_csv)
        # Update to check for UMAP component columns instead of PCA
        if not {'image_filename', 'umap1', 'umap2'}.issubset(point_cloud_df.columns):
            print(f"Error: Missing expected columns in point cloud CSV for epoch {selected_epoch}.")
            point_cloud_df = pd.DataFrame(columns=['image_filename', 'umap1', 'umap2'])
    else:
        print(f"Warning: Point cloud CSV not found for epoch {selected_epoch}.")
        point_cloud_df = pd.DataFrame(columns=['image_filename', 'umap1', 'umap2'])

    # Create point cloud plot
    point_cloud_fig = go.Figure()
    if not point_cloud_df.empty:
        point_cloud_fig.add_trace(go.Scatter(
            x=point_cloud_df['umap1'],
            y=point_cloud_df['umap2'],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            customdata=point_cloud_df['image_filename'],
            name='Generated Points',
            hoverinfo='text',
            hovertext=point_cloud_df['image_filename']
        ))
    else:
        point_cloud_fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='No Data',
            text=['No data available'],
            hoverinfo='text'
        ))
    point_cloud_fig.update_layout(
        title='Point Cloud',
        clickmode='event+select',
        xaxis_title='UMAP Component 1',  # Update label to reflect UMAP
        yaxis_title='UMAP Component 2',  # Update label to reflect UMAP
        hovermode='closest'
    )

    # Load persistence diagram data
    persistence_csv = epoch_to_persistence_csv.get(selected_epoch, '')
    if os.path.exists(persistence_csv):
        persistence_df = pd.read_csv(persistence_csv)
        if not {'birth', 'death', 'dimension', 'image_filename'}.issubset(persistence_df.columns):
            print(f"Error: Missing expected columns in persistence diagram CSV for epoch {selected_epoch}.")
            persistence_df = pd.DataFrame(columns=['birth', 'death', 'dimension', 'image_filename'])
        
        # Filter out infinite deaths if necessary
        persistence_df = persistence_df[persistence_df['death'].notnull()]

        # Create persistence diagram plot
        persistence_fig = go.Figure()

        for dim in persistence_df['dimension'].unique():
            df_dim = persistence_df[persistence_df['dimension'] == dim]
            persistence_fig.add_trace(go.Scatter(
                x=df_dim['birth'],
                y=df_dim['death'],
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                name=f'Dimension {dim}',
                customdata=df_dim['image_filename'],
                hoverinfo='text',
                hovertext=[
                    f'Dimension: {dim}<br>Birth: {b:.2f}<br>Death: {d:.2f}'
                    for b, d in zip(df_dim['birth'], df_dim['death'])
                ]
            ))

        # Add diagonal
        max_val = max(persistence_df['death'].max(), persistence_df['birth'].max())
        persistence_fig.add_shape(
            type='line',
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(dash='dash', color='grey')
        )
        
        persistence_fig.update_layout(
            title='Persistence Diagram',
            clickmode='event+select',
            xaxis_title='Birth',
            yaxis_title='Death',
            hovermode='closest'
        )
    else:
        print(f"Warning: Persistence diagram CSV not found for epoch {selected_epoch}.")
        persistence_fig = go.Figure()
        persistence_fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=8, color='grey')))
        persistence_fig.update_layout(title='Persistence Diagram (No Data)')

    # Handle selected images in either point cloud or persistence diagram
    selected_filenames = []

    if selected_point_cloud_data:
        selected_filenames += [point['customdata'] for point in selected_point_cloud_data['points']]

    if selected_persistence_data:
        selected_filenames += [point['customdata'] for point in selected_persistence_data['points']]

    selected_filenames = list(set(selected_filenames))  # Remove duplicates

    # Display only selected images
    images = []
    for filename in selected_filenames:
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            src = f'data:image/png;base64,{encoded_image}'
            images.append(html.Img(src=src, style={'width': '200px', 'height': '200px', 'margin': '10px'}))

    # Create loss graph figure
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        x=loss_data['Epoch'],
        y=loss_data['G_loss'],
        mode='lines',
        name='Generator Loss',
        line=dict(color='blue')
    ))
    loss_fig.add_trace(go.Scatter(
        x=loss_data['Epoch'],
        y=loss_data['D_loss'],
        mode='lines',
        name='Discriminator Loss',
        line=dict(color='red')
    ))

    loss_fig.update_layout(
        title="Generator and Discriminator Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Losses"
    )

    # Create intrinsic dimension graph
    intrinsic_dim_fig = go.Figure()
    intrinsic_dim_fig.add_trace(go.Scatter(
        x=intrinsic_dim_data['Epoch'],
        y=intrinsic_dim_data['Real_Intrinsic_Dim'],
        mode='lines',
        name='Real Data Intrinsic Dim',
        line=dict(color='green')
    ))
    intrinsic_dim_fig.add_trace(go.Scatter(
        x=intrinsic_dim_data['Epoch'],
        y=intrinsic_dim_data['Generated_Intrinsic_Dim'],
        mode='lines',
        name='Generated Data Intrinsic Dim',
        line=dict(color='orange')
    ))

    intrinsic_dim_fig.update_layout(
        title="Intrinsic Dimensions of Real and Generated Data",
        xaxis_title="Epoch",
        yaxis_title="Intrinsic Dimension",
        legend_title="Intrinsic Dimensions"
    )

    return point_cloud_fig, persistence_fig, images, epoch_text, loss_fig, intrinsic_dim_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
