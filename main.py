import cv2
import numpy as np

def grabcut_algorithm(original_image, bounding_box):
    segment = np.zeros(original_image.shape[:2], np.uint8)
    
    x, y, width, height = bounding_box
    segment[y:y+height, x:x+width] = 1

    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(original_image, segment, bounding_box, background_mdl, foreground_mdl, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"Error en cv2.grabCut: {e}")
        return
    
    new_mask = np.where((segment == 2) | (segment == 0), 0, 1).astype('uint8')
    
    segmented_image = original_image * new_mask[:, :, np.newaxis]
    
    # Encontrar contornos de la región segmentada
    contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Obtener las coordenadas de los puntos en los contornos con aproximación
    segmented_pixels = []
    for contour in contours:
        # Aproximar el contorno poligonalmente
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        for point in approx:
            x, y = point[0]
            segmented_pixels.append({'x': x, 'y': y})
    
    # Mostrar la imagen segmentada
    cv2.imshow('Result', segmented_image)
    
    # Imprimir las coordenadas de los puntos que rodean el segmento
    print("Coordenadas de los puntos que rodean el segmento:")
    print(segmented_pixels)

def main():
    original_image = cv2.imread("images/2.jpg")
    
    # Proporcionar un array de objetos con las coordenadas
    bounding_box_coords = [
        {'x': 0, 'y': 0},  # Top-left corner
        {'x': 673, 'y': 2},  # Top-right corner
        {'x': 0, 'y': 532},  # Bottom-left corner
        {'x': 673, 'y': 532}  # Bottom-right corner
    ]
    
    # Determinar el bounding box a partir de las coordenadas proporcionadas
    x_min = min(coord['x'] for coord in bounding_box_coords)
    y_min = min(coord['y'] for coord in bounding_box_coords)
    x_max = max(coord['x'] for coord in bounding_box_coords)
    y_max = max(coord['y'] for coord in bounding_box_coords)
    
    bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)
    print(f"Bounding box: {bounding_box}")
    
    grabcut_algorithm(original_image, bounding_box)
    
    # Esperar a que el usuario presione una tecla para cerrar la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
