from mmpose.apis import MMPoseInferencer


img_path = 'input/top-section.mp4'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')


# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=False, out_dir="output", num_instances=1,
                               thickness=4, use_oks_tracking=True, draw_heatmap=True)
results = [result for result in result_generator]
