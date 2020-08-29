from PIL import Image, ImageDraw
import numpy as np
import random
import torch
from torch.autograd import Variable
import numpy as np
import os
from torchvision import models, transforms
from scipy.optimize import basinhopping

# backbone_name = 'resnet101'
#
#
#
#
#
#
# model = models.__dict__[backbone_name](pretrained=True)  # N x 2048
# model.eval()
# if torch.cuda.is_available():
#     model.cuda()
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def add_watermark_to_image(image, xs, watermark, sl):
    rgba_image = image.convert('RGBA')
    rgba_watermark = watermark.convert('RGBA')

    image_x, image_y = rgba_image.size
    watermark_x, watermark_y = rgba_watermark.size

    # 缩放图片
    scale = sl
    watermark_scale = min(image_x / (scale * watermark_x), image_y / (scale * watermark_y))
    new_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))
    # rgba_watermark = rgba_watermark.resize(new_size)
    rgba_watermark = rgba_watermark.resize(new_size, resample=Image.ANTIALIAS)
    # 透明度
    rgba_watermark_mask = rgba_watermark.convert("L").point(lambda x: min(x, int(xs[0])))
    rgba_watermark.putalpha(rgba_watermark_mask)

    watermark_x, watermark_y = rgba_watermark.size
    # 水印位置
    # rgba_image.paste(rgba_watermark, (0, 0), rgba_watermark_mask) #右下角
    ##限制水印位置

    a = np.array(xs[1])
    a = np.clip(a, 0, 224 - watermark_x)

    b = np.array(xs[2])
    b = np.clip(b, 0, 224 - watermark_y)

    x_pos = int(a)
    y_pos = int(b)
    rgba_image.paste(rgba_watermark, (x_pos, y_pos), rgba_watermark_mask)  # 右上角
    #rgba_watermark.save('newlogo.png')
    return rgba_image


def label_model(model,input):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    input = transform(input)
    input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
    return model(input.cuda())


def predict_classes(model,img, xs, watermark, target_class, sl):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = add_watermark_to_image(img, xs, watermark, sl)
    imgs_perturbed = imgs_perturbed.convert('RGB')

    predictions = label_model(model,imgs_perturbed).cpu().detach().numpy()
    # This function should always be minimized, so return its complement if needed
    predictions = predictions[0][target_class]
    return predictions






def attack_success(model,img, xs, watermark, sl, target_class, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = add_watermark_to_image(img, xs, watermark, sl)
    attack_image = attack_image.convert('RGB')
    predict = label_model(model,attack_image).cpu().detach().numpy()

    predicted_class = np.argmax(predict)


    if verbose:
        print('Confidence:', predict[0][target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):

        return True




def attack(model,im_before, label, im_watermark, sl, xs=[100, 0, 0],niter=1):
    watermark_x1, watermark_y1 = im_watermark.size
    watermark_scale = min(224 / (sl * watermark_x1), 224 / (sl * watermark_y1))
    # print(watermark_scale)
    watermark_x1 = int(watermark_x1 * watermark_scale)
    watermark_y1 = int(watermark_y1 * watermark_scale)

    def predict_fn(xs):
        return predict_classes(model,im_before, xs, im_watermark, int(label), sl)

    def callback_fn(xs, f, accept):
        return attack_success(model,im_before, xs, im_watermark, sl, int(label), verbose=False)

    class MyTakeStep(object):
        def __init__(self, stepsize=10):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            x[0] += np.random.uniform(-2 * s, 2 * s)
            x[1] += np.random.uniform(-5 * s, 5 * s)
            x[2] += np.random.uniform(-5 * s, 5* s)
            return x

    mytakestep = MyTakeStep()

    class MyBounds(object):
        def __init__(self, xmax=[255, 224 - watermark_x1, 224 - watermark_y1], xmin=[100, 0, 0]):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    mybounds = MyBounds()

    attack_result = basinhopping(func=predict_fn, x0=xs, callback=callback_fn, take_step=mytakestep,
                                 accept_test=mybounds, niter=niter)
    return attack_result






def BH_Calculation(model,img, label, im_watermark, scale, xs=[100, 0, 0],niter=1):
    ori_predict = label_model(model,img).cpu().detach().numpy()
    # print(np.argmax(predict))
    ori_image = add_watermark_to_image(img, xs, im_watermark, scale).convert('RGB')
    predict = label_model(model,ori_image).cpu().detach().numpy()
    Optimal_solutions = []
    result_predict = 100
    if np.argmax(ori_predict) == int(label):
        result = attack(model,img, label, im_watermark, scale, xs=xs,niter=niter)
        print(result.x)
        result_img = add_watermark_to_image(img, result.x, im_watermark, scale)
        result_img = result_img.convert('RGB')
        new_predict = label_model(model,result_img).cpu().detach().numpy()
        if predict[0][int(label)] > new_predict[0][int(label)]:
            Optimal_solutions = result.x
            result_predict = new_predict[0][int(label)]
        else:
            Optimal_solutions = xs
            result_predict = predict[0][int(label)]

    return Optimal_solutions, result_predict


