from tensorflow.keras.models import load_model

def print_detailed_layers(model_path):
    model = load_model(model_path)
    mobilenet_model = model.get_layer('mobilenetv2_1.00_128')
    mobilenet_model.summary()  # Print detailed summary of MobileNetV2

    # Print the layer names and types
    for layer in mobilenet_model.layers:
        print(f"Layer name: {layer.name}, Layer type: {type(layer).__name__}")

if __name__ == "__main__":
    model_path = 'plant_disease_model.h5'
    print_detailed_layers(model_path)

