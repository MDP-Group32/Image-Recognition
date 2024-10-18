import os
import imagezmq
import cv2
import image_recog_script
from Client import PCClient
from PIL import Image
import matplotlib.pyplot as plt
import time

class ImageReceiver:
    def __init__(self):
        # Initialize ImageHub to receive images
        # self.image_hub = imagezmq.ImageHub(open_port="tcp://192.168.32.14:5555")
        print('init')
        self.image_hub = imagezmq.ImageHub(open_port="tcp://*:5555") # --> update on actual day to ensure 1-to-1 receiving
        # self.pc_client = PCClient(ip="192.168.32.1", port=5000)  # Use the RPi's IP
        self.saved_image_paths = [] # --> to store the image path of saved images
        print('init finish')

    def get_model_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print('Current dir: ', current_dir)
        
        # Construct the path to the "images" directory one level above
        # model_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'image_recog', 'best.pt'))
        # model_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'best.pt'))
        model_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'task2v3.pt')) # --> running on the version 3 model (25 Sep 24)
        print('Model dir: ', model_dir)
        return model_dir

    def get_save_directory(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print('Current dir: ', current_dir)
        
        # Construct the path to the "images" directory one level above
        save_dir = os.path.abspath(os.path.join(current_dir, '..', 'images'))
        print('Save dir: ', save_dir)
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        return save_dir

        # Specify the image file name
        # image_file = 'image1.png'
        
        # Construct the full paths to the source and destination files
        # src_path = os.path.join(current_dir, image_file)
        # dst_path = os.path.join(save_dir, image_file)
        # return dst_path
        # Check if the source file exists
        # if os.path.isfile(src_path):
        #     # Copy the image file to the save directory
        #     shutil.copy(src_path, dst_path)
        #     print(f'{image_file} saved successfully.')
        # else:
        #     print(f'{image_file} not found in the current directory.')

    def receive_image(self):
        print('Initiating Image Recognition')
        image_processed = 0
        labels_to_send = []
        while image_processed < 2:
            try:
                    print('Waiting to receive image from rpi')
                    # Receiving image from rpi
                    metadata, image = self.image_hub.recv_image()
                    print('here')
                    data = metadata.split(': ')
                    rpi_name = data[0]
                    image_id = data[1]
                    print(f"Received image {image_id} from {rpi_name}")
                    image_processed += 1

                    # # Receiving obstacle ID from rpi
                    # if self.pc_client.connect():
                    #     obstacle_id = self.pc_client.receive()
                    #     print(f"Received image from {rpi_name} for {obstacle_id}")
                    # else:
                    #     print("Obstacle ID not received!")

                    # Display the image using OpenCV
                    cv2.imshow(f"Image from {rpi_name}", image)
                    cv2.waitKey(1)
                    save_dir = self.get_save_directory()

                    # Construct the full path to save the image
                    image_filename = os.path.join(save_dir, f"{image_id}.jpg")
                    cv2.imwrite(image_filename, image)
                    print(f"Image saved to {image_filename}")
                    print('Model path: ', self.get_model_path())

                    # Send a reply to acknowledge receipt
                    # self.image_hub.send_reply(b'Image received')
                    # image_filename = os.path.join(save_dir, f"raspberrypi.jpg")

                    # Image recognition and result handling 
                    model = image_recog_script.load_model(self.get_model_path())
                    print('model loaded')
                    labels, annotatedImage = image_recog_script.predict_image(image_filename, model)
                    print('reached here')

                    right = 'rightarrow_id38'
                    left = 'leftarrow_id39'
                    default_label = right # change here the default direction

                    # Strategy 1: current default
                    if len(labels) == 0: # Handling failure to recognise image
                        print("Opting for default direction")
                        label = self.convert_label_to_output(default_label)
                        annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                        annotatedImage.save(annotated_image_path)
                        self.saved_image_paths.append((image_id, annotated_image_path))
                        print(f"Annotated image saved to {annotated_image_path}")
                        result = label.encode('utf-8')
                        self.image_hub.send_reply(result) 
                        print(f"Sent reply: {result}")

                    found = False  # Flag to check if 'left' or 'right' has been found
                    for label in labels:  # For single recognition (ranked by proximity to RPi cam)
                        if label == 'leftarrow_id39' or label == 'rightarrow_id38':
                            # If 'left' or 'right' is found, process the label and exit the loop
                            label = self.convert_label_to_output(label)
                            annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                            annotatedImage.save(annotated_image_path)
                            self.saved_image_paths.append((image_id, annotated_image_path))
                            print(f"Annotated image saved to {annotated_image_path}")
                            result = label.encode('utf-8')
                            self.image_hub.send_reply(result) 
                            print(f"Sent reply: {result}")
                            found = True
                            break

                    if not found:
                        print("Opting for default direction")
                        label = self.convert_label_to_output(default_label)
                        annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                        annotatedImage.save(annotated_image_path)
                        self.saved_image_paths.append((image_id, annotated_image_path))
                        print(f"Annotated image saved to {annotated_image_path}")
                        result = label.encode('utf-8')
                        self.image_hub.send_reply(result) 
                        print(f"Sent reply: {result}")

                    #-----------------------------------------------------------------------------------------------------------

                    # # Strategy 2: Use 1st image recognition as back up
                    # if image_processed == 1:
                    #     found = False
                    #     if len(labels) == 0: # Handling failure to recognise image
                    #         print("Opting for default direction")
                    #         label = self.convert_label_to_output(default_label)
                    #         annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                    #         annotatedImage.save(annotated_image_path)
                    #         self.saved_image_paths.append((image_id, annotated_image_path))
                    #         print(f"Annotated image saved to {annotated_image_path}")
                    #         result = label.encode('utf-8')
                    #         self.image_hub.send_reply(result) 
                    #         print(f"Sent reply: {result}")

                    #     for label in labels: # For dual shot with first image as back up
                    #         if label == 'leftarrow_id39' or label == 'rightarrow_id38':
                    #             label = self.convert_label_to_output(label)
                    #             labels_to_send.append(label)
                    #             found = True
                    #             if len(labels_to_send) == 2:
                    #                 break
                    #     label = labels_to_send[0]
                    #     if not found:
                    #             print("Opting for default direction")
                    #             label = self.convert_label_to_output(default_label)
                    #             annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                    #             annotatedImage.save(annotated_image_path)
                    #             self.saved_image_paths.append((image_id, annotated_image_path))
                    #             print(f"Annotated image saved to {annotated_image_path}")
                    #             result = label.encode('utf-8')
                    #             self.image_hub.send_reply(result) 
                    #             print(f"Sent reply: {result}")
                    #     annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                    #     annotatedImage.save(annotated_image_path)
                    #     self.saved_image_paths.append((image_id, annotated_image_path))
                    #     print(f"Annotated image saved to {annotated_image_path}")
                    #     result = label.encode('utf-8')
                    #     self.image_hub.send_reply(result) 
                    #     print(f"Sent reply: {result}")
                    #     break

                    # if image_processed == 2:
                    #     if len(labels) == 0:
                    #         if len(labels_to_send) == 0:
                    #             label = self.convert_label_to_output(default_label)
                    #             break
                    #         label = labels_to_send[0]
                    #     else:
                    #         for label in labels:
                    #             if label == 'leftarrow_id39' or label == 'rightarrow_id38':
                    #                 label = self.convert_label_to_output(label)
                    #                 break
                    #     annotated_image_path = os.path.join(save_dir, f"{image_id}_{label}_annotated.jpg")
                    #     annotatedImage.save(annotated_image_path)
                    #     self.saved_image_paths.append((image_id, annotated_image_path))
                    #     print(f"Annotated image saved to {annotated_image_path}")
                    #     result = label.encode('utf-8')
                    #     self.image_hub.send_reply(result) 
                    #     print(f"Sent reply: {result}")
                    #     break
                        
                    #-----------------------------------------------------------------------------------------------------------

                    # Strategy 3: Single Recognition (not recommended with lower accuracy)
                    # for label in labels:  # For single shot recognition
                    #     if label == 'leftarrow_id39' or label == 'rightarrow_id38':
                    #         # If 'left' or 'right' is found, process the label
                    #         label = self.convert_label_to_output(label)
                            
                    #         # Add the label to the list to send
                    #         labels_to_send.append(label)
                            
                    #         # Check if we have found two arrows
                    #         if len(labels_to_send) == 2:
                    #             # Create the result string from the two labels
                    #             result = "".join(labels_to_send)
                    #             # Annotate and save the image using the result as part of the file name
                    #             annotated_image_path = os.path.join(save_dir, f"{image_id}_{result}_annotated.jpg")
                    #             annotatedImage.save(annotated_image_path)
                    #             self.saved_image_paths.append((image_id, annotated_image_path))
                    #             print(f"Annotated image saved to {annotated_image_path}")
                                
                    #             # Send the result back
                    #             self.image_hub.send_reply(result.encode('utf-8'))
                    #             print(f"Sent reply: {result}")
                    #             break  # Exit the loop after sending the first two arrows

                    


            # except KeyboardInterrupt:
            #     print("KeyboardInterrupt: Stopping image reception.")
            #     self.stitch_images()

            except Exception as e:
                image_expected = 0
                print(f"Image Recognition Terminated: {e}")
                # break

        # self.image_hub.close()
        # self.send_obstacle_data()
        self.stitch_images()

    def stitch_images(self):
        # Sort the saved images by obstacle ID (image name)
        self.saved_image_paths.sort(key=lambda x: x[0])  # Sorting by the image name (obstacle ID)

        images = [Image.open(path) for _, path in self.saved_image_paths]
        # obstacle_ids = [obstacle_id for obstacle_id, _ in self.saved_image_paths]  # Get obstacle IDs
        num_images = len(images)

        # Create a figure with subplots (one for each image)
        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))  # Adjust size as necessary
        fig.suptitle('MDP Grp 32 Task 2 result', fontsize=16)  # Main heading

        if num_images == 1:
            axes = [axes]

        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img)
            ax.axis('off')
            # obstacle_id = obstacle_ids[i]
            # ax.set_title(f"Obstacle ID: {obstacle_id}", fontsize=10)  # Replace label accordingly

            # Extracting the obstacle ID and the image detection result from the image path
            image_id, image_path = self.saved_image_paths[i]
            detection_result = os.path.basename(image_path).split('_')[1]  # Extract detection result from the file name
            ax.set_title(f"Image ID: {image_id}\nDetection: {detection_result}", fontsize=10)

        plt.tight_layout()
        plt.show()

    def send_obstacle_data(self):
        try:
            pc_client = PCClient(ip="192.168.32.1", port=5000)
            self.saved_image_paths.sort(key=lambda x: x[0])
            formatted_data = "TARGET:" + ";".join([f"{obstacle_id},{os.path.basename(image_path).split('_')[1]}" 
                                    for obstacle_id, image_path in self.saved_image_paths]) # Current Output 'TARGET:ID,Result;ID,Result...'
            while True:
                if pc_client.connect():
                    print(f"Sending obstacle data: {formatted_data}")
                    pc_client.send(formatted_data)
                    print("Data sent successfully.")
                    break
                else: 
                    print("Message communication not esbalished, retrying in 1 second")
                    time.sleep(1)
            pc_client.close()

        except Exception as e:
            print(f"Error while sending obstacle data: {e}")


    
    def convert_label_to_output(self, label):
        print("original label:", label)
        label_to_output = {
            # 'A_id20': 'A',
            # 'B_id21': 'B',
            # 'C_id22': 'C',
            # 'D_id23': 'D',
            # 'E_id24': 'E',
            # 'F_id25': 'F',
            # 'G_id26': 'G',
            # 'H_id27': 'H',
            # 'S_id28': 'S',
            # 'T_id29': 'T',
            # 'U_id30': 'U',
            # 'V_id31': 'V',
            # 'W_id32': 'W',
            # 'X_id33': 'X',
            # 'Y_id34': 'Y',
            # 'Z_id35': 'Z',
            # 'downarrow_id37': 'down',
            # 'eight_id18': '8',
            # 'five_id15': '5',
            # 'four_id14': '4',
            'leftarrow_id39': 'l',
            # 'nine_id19': '9',
            # 'one_id11': '1',
            'rightarrow_id38': 'r'
            # 'seven_id17': '7',
            # 'six_id16': '6',
            # 'stop_id40': 'stop',
            # 'three_id13': '3',
            # 'two_id12': '2',
            # 'uparrow_id36': 'up'
            }
        new_label = label_to_output.get(label,'unknown')
        print("new label: ", new_label)
        return new_label