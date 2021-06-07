import os
import platform
import timeit
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

class File(tf.gfile.GFile):
    # from WAE
    """Wrapper on GFile extending seek, to support what python file supports."""
    def __init__(self, *args):
        super(File, self).__init__(*args)

    def seek(self, position, whence=0):
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)
        
def o_gfile(filename, mode):
    # from WAE
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)

def argv_parse(argvs):
    arglist = [arg.strip() for args in argvs[1:] for arg in args.split('=') if not arg.strip()=='']
    arglist.reverse()
    argdict = dict()
    argname = arglist.pop()
    while len(arglist) > 0:
        #log.debug('argname={}'.format(argname))
        if '--' not in argname:
            raise Exception('python argument error')
        argv = []
        while len(arglist) > 0:
            arg = arglist.pop()
            if '--' in arg:
                argdict[argname.split('--')[-1]] = argv
                argname = arg
                break
            #log.debug('arg={}'.format(arg))
            argv.append(arg)
    argdict[argname.split('--')[-1]] = argv
    return argdict

def file_path_fold(path, fold):
    path = path.split('.')
    return '.'+''.join(path[:-1])+'_'+str(fold)+'.'+path[-1]

def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)
    
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
def print_sysinfo():
    print('\nPython version  : {}'.format(platform.python_version()))
    print('compiler        : {}'.format(platform.python_compiler()))
    print('\nsystem     : {}'.format(platform.system()))
    print('release    : {}'.format(platform.release()))
    print('machine    : {}'.format(platform.machine()))
    print('processor  : {}'.format(platform.processor()))
    print('CPU count  : {}'.format(mp.cpu_count()))
    print('interpreter: {}'.format(platform.architecture()[0]))
    print('\n\n')

####################################################################################################################################

def _set_ax_plot(ax, models, y_names, y_trans, x_name = 'epochs', niters=None, legend_loc=None):
    line_style = ['-',':','-.','--']
    x_title = x_name.title()
    if 'val' in y_names[0]: title = 'Validation History (per %s)' % (x_title)
    else: title = 'History (per %s)' % (x_title)
    ax.set_title(title, fontdict={'fontsize':15})
    
    for col, model in enumerate(models):
        hist_path = './%s/hist.json' % (model)
        with open(hist_path, 'r') as f:
            history = json.load(f)
        max_epoch = np.max([len(v) for v in history.values()])

        if not niters == None: niter = niters[col]
        else: niter = None
        if 'epoch' in x_name: index = np.arange(1, max_epoch+1)
        elif 'iter' in x_name: index = np.arange(1, max_epoch*niter+1, niter)
        else: raise ValueError

        for line, (y_name, yt) in enumerate(zip(y_names, y_trans)):
            value = yt(np.array(history[y_name]))
            if 'val' in y_name: y_title = "_".join(y_name.split('_')[1:]).title()
            else: y_title = y_name.title()
            ax.plot(index, value, 'C%s.%s' % (col+1,line_style[line]), label='%s-%s'% (model, y_title), alpha=1.)
            if 'wae_loss' in y_name : # total epoch min
                ax.plot(index[np.argmin(value)], np.min(value), 'C%s*' % (col+1), markersize=12)
                ax.annotate("%.3f" % np.min(value), (index[np.argmin(value)],np.min(value))) 
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.legend(loc=legend_loc, fontsize='small') #'medium')

def plot_history(models, history_types = ['validation','train'],
                 y_names_list = [['wae_loss','discriminator_gan_loss'], ['wae_penalty','wae_reconstruction','sharpness']],
                 y_transform = [[lambda x: x]*2, [lambda x: np.log(np.abs(x))]*3],
                 x_name='epochs', figsize=(20,20), niters=None):
    fig, axes = plt.subplots(len(y_names_list), len(history_types), figsize=figsize)
    if len(y_names_list) == 1: axes = np.expand_dims(axes, 0)
    if len(history_types) == 1: axes = np.expand_dims(axes, -1)
    for j in range(len(history_types)):
        for i, y_names in enumerate(y_names_list):
            if 'val' in history_types[j]: _set_ax_plot(axes[i,j], models, ['val_%s' % yn for yn in y_names], y_transform[i], x_name, niters)
            else: _set_ax_plot(axes[i,j], models, y_names, y_transform[i], x_name, niters)
    fig.tight_layout()
    
def plot_images(x, title=None, cmap=None):
    nrows = np.ceil(np.sqrt(x.shape[0])).astype(np.int)
    ncols = x.shape[0]//nrows
    if nrows * ncols!= x.shape[0]: nrows += 1

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10,10))
    fig.suptitle(title)
    for i in range(nrows):
        for j in range(ncols):
            if i*ncols + j < x.shape[0]:
                if cmap != None: axes[i][j].imshow(x[i*ncols + j,:,:], cmap=cmap)
                else: axes[i][j].imshow(0.5*x[i*ncols + j,:,:]+0.5)
                axes[i][j].axis('off')
            # axes[i][j].set_xticklabels([])
            # axes[i][j].set_yticklabels([])
    fig.subplots_adjust(wspace=0.025, hspace=0.05)
    return fig
    
def plot_reconstruction(x, recon_x, sharpness=None, savename=None, cmap=None):
    if sharpness is not None:
        plot_images(x, title='Original images with sharpness %.5f' % sharpness[0], cmap=cmap)
        if savename != None: plt.savefig('%s_original.png'%savename)
        plot_images(recon_x, title='Reconstruction images with sharpness %.5f' % sharpness[1], cmap=cmap)
        if savename != None: plt.savefig('%s_recon.png'%savename)
    else:
        plot_images(x, title='Original images')
        if savename != None: plt.savefig('%s_original.png'%savename)
        plot_images(recon_x, title='Reconstruction images', cmap=cmap)
        if savename != None: plt.savefig('%s_recon.png'%savename)
