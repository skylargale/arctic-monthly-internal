#!/usr/bin/env python
# coding: utf-8

# Import libraries
import os
import glob
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import argparse

# Configure environment - CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Months
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


def weighted_spatial(data_array, lats):
    '''Compute weighted spatial averages over lat/lon dims.'''
    weights = np.cos(np.deg2rad(lats))
    weighted_data = data_array * weights[:, np.newaxis]
    weighted_sum = np.nansum(weighted_data)
    denom = np.nansum(weights) * data_array.shape[1]
    return weighted_sum / denom


def get_data(paths):
    '''Load data efficiently.'''
    X_MAPS, Y_DATA, model_names = [], [], []
    for path in paths:
        model_names.append(os.path.splitext(os.path.basename(path))[0])

        with xr.open_dataset(path, chunks={'time': 100}) as ds:
            ds_sel = ds.sel(period=slice(1900, 2050))
            if SAT_SLP:
                sat = ds_sel.X_SAT_SLP[..., 0].values
                slp = ds_sel.X_SAT_SLP[..., 1].values
                X_maps = np.stack([sat, slp], axis=-1)
            else:
                X_maps = ds_sel.X_SAT_SLP[..., 1].values

            Y_data = ds_sel.Y_I_E_SUM.values

        X_MAPS.append(X_maps)
        Y_DATA.append(Y_data)

    return X_MAPS, Y_DATA, model_names


def compute_actual_trends(obs_maps):
    '''Compute observed actual trends.

    Arctic-specific: averages over the SLP-product axis and restricts the weighted
    mean to the northernmost 8 rows of the Arctic-cropped grid (roughly 60N+),
    consistent with the "actual" scalar trend reported in the manuscript.
    '''
    lats = np.arange(-88.75, 88.751, 2.5)

    actual_trends_all_obs = []
    for SAT in range(4):
        actual_trends_one_obs = []
        for SLP in range(3):
            actual_trends = []
            for month in range(12):
                actual_trends.append(weighted_spatial(obs_maps[SAT, SLP, month, 20:, :, 0], lats[64:]))
            actual_trends_one_obs.append(actual_trends)
        actual_trends_all_obs.append(actual_trends_one_obs)

    actual_trends_obs_means = np.nanmean(actual_trends_all_obs, axis=1)

    print(f'Observed {months[MONTH_IDX]} mean: {np.nanmean(actual_trends_obs_means[:, MONTH_IDX], axis=0):.3f} K/dec')

    return actual_trends_obs_means, obs_maps


def create_CNN(input_shape):
    '''Initialize CNN.'''
    inpt = layers.Input(shape=input_shape)

    # Convolutional layer
    conv = layers.Conv2D(16, (1, 3), padding='same', activation='relu')(inpt)
    pool = layers.MaxPooling2D(1)(conv)
    drop = layers.Dropout(0.5)(pool)

    # Flatten and output layer
    flat = layers.Flatten()(drop)
    out = layers.Dense(2)(flat)

    model = models.Model(inputs=inpt, outputs=out)
    return model


def geometry_weight(lats, mode):
    '''Latitude-dependent multiplicative weight for the geometry sensitivity test.

    Multiplies each row of the input maps by a latitude-dependent weight before it
    reaches the CNN. This directly tests the reviewer's concern that grid cells near
    the pole cover less true area than cells at lower latitudes and are therefore
    over-represented in an unweighted convolution. Compare a run with
    GEOMETRY_WEIGHTING = None against one with 'cos' (or 'sqrt_cos') to see how much
    the CNN's learned Arctic-mean trend estimate shifts.'''
    if mode is None:
        return np.ones_like(lats, dtype=np.float32)
    lat_rad = np.deg2rad(lats)
    if mode == 'cos':
        w = np.cos(lat_rad)
    elif mode == 'sqrt_cos':
        w = np.sqrt(np.cos(lat_rad))
    else:
        raise ValueError(f"Unknown GEOMETRY_WEIGHTING mode: {mode!r}")
    w = w / np.mean(w)  # keep overall input amplitude comparable to the unweighted case
    return w.astype(np.float32)


def apply_geometry_weighting(X, lats, mode, has_channel=None):
    '''Multiply an array whose spatial dims are (..., lat, lon) or (..., lat, lon, channel)
    by a latitude-dependent weight, broadcasting over any number of leading dims
    (ensemble, period, month, ...).'''
    if mode is None:
        return X
    if has_channel is None:
        has_channel = SAT_SLP
    X = np.asarray(X)
    w = geometry_weight(lats, mode)
    if has_channel:
        w_shape = (1,) * (X.ndim - 3) + (len(lats), 1, 1)
    else:
        w_shape = (1,) * (X.ndim - 2) + (len(lats), 1)
    return X * w.reshape(w_shape)


def prepare_data(splice_X, splice_Y, hist_X, hist_Y, cv, m):
    '''Bootstrap 10 ensemble members to train on from models with more than 10 members.

    `m` is the month index used for both the spliced and historical training data.
    Kept as a plain parameter (rather than reading MONTH_IDX directly) so the same
    function can be reused for a different month without editing its body.
    '''
    splice_X_10ens, splice_Y_10ens = [], []
    for idx in range(len(splice_X)):
        random_idxs = tf.random.shuffle(tf.range(tf.shape(splice_X[idx])[0]))[:10]
        splice_X_10ens.append(tf.gather(splice_X[idx][:, :, m], random_idxs))
        splice_Y_10ens.append(tf.gather(splice_Y[idx][:, :, m], random_idxs))

    hist_X_10ens, hist_Y_10ens = [], []
    for idx in range(len(hist_X)):
        random_idxs = tf.random.shuffle(tf.range(tf.shape(hist_X[idx])[0]))[:10]
        hist_X_10ens.append(tf.gather(hist_X[idx][:, :, m], random_idxs))
        hist_Y_10ens.append(tf.gather(hist_Y[idx][:, :, m], random_idxs))

    # Delete the specified index and concatenate, specify for E3SM-2-0
    if no_E3SM and cv == 0:
        X_train_spliced = tf.concat([splice_X_10ens[i] for i in range(len(splice_X_10ens))], axis=0)
        Y_train_spliced = tf.concat([splice_Y_10ens[i] for i in range(len(splice_Y_10ens))], axis=0)
    else:
        X_train_spliced = tf.concat([splice_X_10ens[i] for i in range(len(splice_X_10ens)) if i != cv], axis=0)
        Y_train_spliced = tf.concat([splice_Y_10ens[i] for i in range(len(splice_Y_10ens)) if i != cv], axis=0)

    # Concatenate the hist tensors
    X_train_hist = tf.concat(hist_X_10ens, axis=0)
    Y_train_hist = tf.concat(hist_Y_10ens, axis=0)

    # Reshape to flatten first two dims (ensemble + time)
    def reshape_flat(tensor):
        shape = tf.shape(tensor)
        new_shape = tf.concat([[-1], shape[2:]], axis=0)
        return tf.reshape(tensor, new_shape)

    X_train_spliced = reshape_flat(X_train_spliced)
    X_train_hist = reshape_flat(X_train_hist)

    Y_train_spliced = tf.reshape(Y_train_spliced, [-1, tf.shape(Y_train_spliced)[2]])
    Y_train_hist = tf.reshape(Y_train_hist, [-1, tf.shape(Y_train_hist)[2]])

    X_train = tf.concat([X_train_spliced, X_train_hist], axis=0)
    Y_train = tf.concat([Y_train_spliced, Y_train_hist], axis=0)

    return X_train, Y_train


def reinitialize_weights(model):
    '''Re-randomize a model's weights in place using each layer's own initializers.

    Building and compiling a fresh Keras model for every one of the 400
    (8 folds x 50 randomizations) training runs forces TensorFlow to retrace the
    training graph from scratch each time, which dominates runtime on CPU for a
    model this small. Reusing one compiled model and just re-randomizing its
    weights keeps the same 50-random-initializations design but traces the graph
    once instead of 400 times.

    Reuses each layer's initializer *class* but constructs a fresh instance with a
    new random seed for every call - reusing the same initializer *instance*
    (as an earlier version of this function did) makes Keras return identical
    weights on every call, which would silently turn all 50 "randomizations"
    into copies of the same model.
    '''
    def fresh_call(initializer, shape, dtype):
        init_cls = type(initializer)
        try:
            new_initializer = init_cls(seed=np.random.randint(0, 2**31 - 1))
        except TypeError:
            new_initializer = init_cls()
        return new_initializer(shape=shape, dtype=dtype)

    for layer in model.layers:
        if getattr(layer, 'kernel', None) is not None:
            layer.kernel.assign(fresh_call(layer.kernel_initializer, layer.kernel.shape, layer.kernel.dtype))
        if getattr(layer, 'bias', None) is not None:
            layer.bias.assign(fresh_call(layer.bias_initializer, layer.bias.shape, layer.bias.dtype))


# Configuration
region = 'Arctic'
no_E3SM = True
SAT_SLP = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "--month",
    choices=["march", "april", "may"],
    required=True,
    help="Month to run."
)

parser.add_argument(
    "--geometry",
    choices=["none", "cos", "sqrt_cos"],
    default="none",
    help="Geometry weighting."
)

args = parser.parse_args()

month_lookup = {
    "march": 2,
    "april": 3,
    "may": 4,
}

MONTH_IDX = month_lookup[args.month]

GEOMETRY_WEIGHTING = None if args.geometry == "none" else args.geometry

print(f"Running month = {args.month}")
print(f"Geometry weighting = {GEOMETRY_WEIGHTING}")

lats_full = np.arange(-88.75, 88.751, 2.5)
lats_arctic = lats_full[-28:]

print('Region:', region, '\nMonth:', months[MONTH_IDX], '\nGeometry weighting:', GEOMETRY_WEIGHTING)

# Load observation data
sat_paths = np.sort(glob.glob('../data/training-data/monthly/observations/sat/*.nc'))[::-1]
slp_paths = glob.glob('../data/training-data/monthly/observations/slp/*')

obs_maps = []
for sat_path in sat_paths:
    ds_sat = xr.load_dataset(sat_path)
    X_maps_sat = ds_sat.DATA[:, 44:]

    for slp_path in slp_paths:
        ds_slp = xr.load_dataset(slp_path)
        X_maps_slp = ds_slp.DATA[:, 44:]
        obs_maps.append(np.stack([X_maps_sat, X_maps_slp], axis=-1))

obs_maps = np.reshape(obs_maps, (4, 3, 12, 28, 144, 2))
actual_trends_obs_means, obs_maps = compute_actual_trends(obs_maps)
obs_maps = np.reshape(obs_maps, (12, 12, 28, 144, 2))

# Sensitivity test: apply the same geometry weighting used on the model inputs
obs_maps = apply_geometry_weighting(obs_maps, lats_arctic, GEOMETRY_WEIGHTING)

# Load simulated data
spliced_path = '../data/training-data/monthly/spliced/'
hist_path = '../data/training-data/monthly/hist/'
spliced = glob.glob(spliced_path+region.lower()+'/*')
hist = glob.glob(hist_path+region.lower()+'/*')

# Remove unwanted files
spliced = [model for model in spliced if 'OthersAllEM.nc' not in model]
hist = [model for model in hist if 'OthersAllEM.nc' not in model]

# Store data
splice_X, splice_Y, model_names_spliced_old = get_data(spliced)
hist_X, hist_Y, model_names_hist = get_data(hist)

# Reorganize models by warming indices
models_reorganized = [
    'E3SM-2-0',
    'CanESM5',
    'MIROC6',
    'MPI-ESM1-2-LR',
    'IPSL-CM6A-LR',
    'ACCESS-ESM1-5',
    'CESM2',
    'CESM2_SMBB'
]

ranked_warming_indices = [model_names_spliced_old.index(name) for name in models_reorganized]
splice_X_array = [splice_X[i] for i in ranked_warming_indices]
splice_Y_array = [splice_Y[i] for i in ranked_warming_indices]
model_names_spliced = [model_names_spliced_old[i] for i in ranked_warming_indices]

# Sensitivity test: apply the same geometry weighting to model input maps
splice_X_array = [apply_geometry_weighting(x, lats_arctic, GEOMETRY_WEIGHTING) for x in splice_X_array]
hist_X = [apply_geometry_weighting(x, lats_arctic, GEOMETRY_WEIGHTING) for x in hist_X]

# Convert into tensors
splice_X, splice_Y = [], []
[splice_X.append(tf.convert_to_tensor(splice_X_array[i])) for i in range(len(splice_X_array))]
[splice_Y.append(tf.convert_to_tensor(splice_Y_array[i])) for i in range(len(splice_Y_array))]
obs_maps = tf.convert_to_tensor(obs_maps)

# Check data we are using for training
print('\n')
for i, splice in enumerate(splice_X):
    print(i+1, np.shape(splice), model_names_spliced[i])

print('\n')
for i, splice in enumerate(splice_Y):
    print(i+1, np.shape(splice), model_names_spliced[i])

# Remove E3SMv2 from training data
if no_E3SM:
    print('\n==> E3SM not included in training data.')

    splice_X_cut = splice_X[1:]
    splice_Y_cut = splice_Y[1:]
    model_names_spliced_cut = model_names_spliced[1:]

    print('\n')
    for i, splice in enumerate(splice_X_cut):
        print(i+1, np.shape(splice), model_names_spliced_cut[i])

    print('\n')
    for i, splice in enumerate(splice_Y_cut):
        print(i+1, np.shape(splice), model_names_spliced_cut[i])
else:
    print('\n==> E3SM included in training data.')

print('\nDone!')

# Create arrays for which to store result
cv_preds_and_vals = []
cv_obs = []
cv_mse = []
prediction_error = []

# Main training loop
Xscale = MinMaxScaler()
Yscale = MinMaxScaler()

output_dir = f'./preds_and_vals/{months[MONTH_IDX].lower()}_{region.lower()}/'
os.makedirs(output_dir, exist_ok=True)

# Build and compile the model once
model = create_CNN(input_shape=(len(lats_arctic), 144, 2 if SAT_SLP else 1))
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=1e-4))

with tf.device('/CPU:0'):

    # Loop through each large ensemble we test on
    for cv in range(len(splice_X)):
        print('Cross Validation:', model_names_spliced[cv])

        cv_preds_and_vals_one_randomization = []
        cv_obs_one_randomization = []

        # Create the CNN 50 times to cut noise from random initializations
        for randomization in range(50):
            reinitialize_weights(model)

            # Define training data (train and test consistently use MONTH_IDX)
            X_train, Y_train = prepare_data(splice_X_cut, splice_Y_cut, hist_X, hist_Y, cv, MONTH_IDX)

            # Reshape training data for CNN
            X_train_reshaped = tf.reshape(X_train, (tf.shape(X_train)[0], -1))
            Y_train_reshaped = Y_train[:, :2]

            # Define testing data
            X_test = splice_X[cv][:, -6, MONTH_IDX]
            X_test_reshaped = tf.reshape(X_test, (tf.shape(X_test)[0], -1))
            Y_test_reshaped = splice_Y[cv][:, -6, MONTH_IDX, :2]

            # Scale training and testing data
            Xscale.fit(X_train_reshaped.numpy())
            Yscale.fit(Y_train_reshaped.numpy())
            X_train_scaled = Xscale.transform(X_train_reshaped.numpy())
            X_test_scaled = Xscale.transform(X_test_reshaped.numpy())
            Y_train_scaled = Yscale.transform(Y_train_reshaped.numpy())

            # Put data back into sample x map x channel shape
            if SAT_SLP:
                X_train_scaled = tf.reshape(X_train_scaled, tf.shape(X_train))
                X_test_scaled = tf.reshape(X_test_scaled, tf.shape(X_test))
            else:
                X_train_scaled = tf.reshape(X_train_scaled, (*X_train.shape[:3], 1))
                X_test_scaled = tf.reshape(X_test_scaled, (*X_test.shape[:3], 1))

            # Fit the model with validation data
            model.fit(
                X_train_scaled,
                Y_train_scaled,
                epochs=10,
                verbose=0,
                batch_size=32,
                shuffle=True
            )

            # Make predictions on testing data
            Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            Y_preds_unscaled = Yscale.inverse_transform(Y_pred_scaled)

            # Reshape prediction and actual values
            Y_pred = tf.transpose(Y_preds_unscaled).numpy().tolist()
            Y_test_one_model = tf.transpose(Y_test_reshaped).numpy().tolist()

            # Append sum values to prediction and actual values
            Y_pred.append(np.nansum(Y_pred, axis=0))
            Y_test_one_model.append(np.nansum(Y_test_one_model, axis=0))
            cv_preds_and_vals_one_randomization.append([tf.transpose(Y_pred), tf.transpose(Y_test_one_model)])

            # Prepare predicting data
            obs_map = obs_maps[:, MONTH_IDX]
            X_obs_reshaped = tf.reshape(obs_map, (tf.shape(obs_map)[0], -1))

            # Make predictions on observations
            X_obs_scaled = Xscale.transform(X_obs_reshaped.numpy())
            X_obs_map_scaled = tf.reshape(X_obs_scaled, tf.shape(obs_map))
            obs_Y_pred_scaled = model.predict(X_obs_map_scaled, verbose=0)

            # Reshape observation prediction values
            obs_Y_pred_unscaled = Yscale.inverse_transform(obs_Y_pred_scaled)
            Y_obs = tf.transpose(obs_Y_pred_unscaled).numpy().tolist()
            Y_obs.append(np.nansum(Y_obs, axis=0))
            cv_obs_one_randomization.append(Y_obs)

        # Find mean prediction for the current cross-validation fold
        mean_of_randomizations_val = tf.reduce_mean(cv_preds_and_vals_one_randomization, axis=0).numpy()
        mean_of_randomizations_obs = tf.reduce_mean(cv_obs_one_randomization, axis=0).numpy()
        error = mean_of_randomizations_val[0] - mean_of_randomizations_val[1]
        prediction_error.append(error)
        mse_of_cv = [tf.reduce_mean(tf.square(error[:, i])).numpy() for i in range(3)]

        # Append to arrays of all cvs
        cv_preds_and_vals.append(mean_of_randomizations_val)
        cv_obs.append(mean_of_randomizations_obs)
        cv_mse.append(mse_of_cv)

# Save cross validation predicted values
tag = f'_{GEOMETRY_WEIGHTING}' if GEOMETRY_WEIGHTING else ''
for i in range(len(model_names_spliced)):
    np.save(output_dir+region.lower()+'_'+str(model_names_spliced[i])+tag+'.npy', np.array(cv_preds_and_vals[i]))

# Save observation predicted values
np.save(output_dir+region+'_obs'+tag+'.npy', np.array(cv_obs))

print('Done!')

titles = ['Internal', 'External', 'Sum']
obs_labels = ['NOAAv6', 'HadCRUTv5', 'GISTv4', 'BerkeleyEarth']
obs_markers = ['o', 'x', '+', '*']
linestyles = ['-', '--', ':', '-.']
colors = cm.rainbow(np.linspace(0, 1, 8))[::-1]
bbox = dict(facecolor='white', ec='black', boxstyle='round')
ticks = np.arange(-1, 1.01, 0.25),  np.arange(0, 1.51, 0.25), np.arange(0, 2.1, 0.5)

obs_actual = str(np.round(np.nanmean(actual_trends_obs_means, axis=0), 3)[MONTH_IDX])

fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=300, facecolor='white')
cv_preds_and_vals_ = cv_preds_and_vals
cv_obs_ = np.swapaxes(np.array(cv_obs), 0, 2)

new_errors = []
for error in prediction_error:
    squared_error = np.multiply(error, error)
    new_errors.append(squared_error/len(squared_error))
new_errors_all = np.concatenate(new_errors)
new_errors_sum = np.nansum(new_errors_all, 0)
new_errors_std = np.sqrt(new_errors_sum/8)

title_tag = f' ({months[MONTH_IDX]}' + (f', {GEOMETRY_WEIGHTING} weighting)' if GEOMETRY_WEIGHTING else ')')

for i in range(3):
    axs[0].text(0.03, 0.9, f'{region}{title_tag}', fontsize=12, transform=axs[0].transAxes, va='top', ha='left')
    all_obs_spread = np.nanmean(cv_obs_[:, i, :], 1).ravel()
    obs_mean = np.nanmean(all_obs_spread)
    sigma_obs = np.nanstd(all_obs_spread)
    sigma_cnn = new_errors_std[i]
    print('Sigma_obs:', sigma_obs)
    print('Sigma_CNN:', sigma_cnn)
    sigma = np.sqrt(sigma_obs**2+sigma_cnn**2)
    axs[i].axvspan(obs_mean-2*sigma, obs_mean+2*sigma, color='grey', alpha=0.2, ec=None)
    cv_obs_set = cv_obs_[:, i, :].reshape(4, 3, 8)
    mean = str(round(obs_mean, 3))
    std = str(round(sigma, 3))

    if i in [0, 1]:
        text = '$\overline{x}$$_{pred}$: '+mean+' K/dec\nσ: '+std+' K/dec'
        axs[i].text(0.97, 0.03, text, transform=axs[i].transAxes, fontsize=10, va='bottom', ha='right', bbox=bbox)
        for idx in range(len(cv_obs_set)):
            obs_set_all_cv = cv_obs_set[idx]
            mean_obs = np.nanmean(obs_set_all_cv)
            axs[i].plot(np.repeat(mean_obs, 3), [-5, 0.3, 5], linestyle=linestyles[idx], linewidth=1.5, color='k', label=obs_labels[idx])
    else:
        text = '$\overline{x}$$_{actual}$: '+obs_actual+' K/dec'+'\n$\overline{x}$$_{pred}$: '+mean+' K/dec\nσ: '+std+' K/dec'
        axs[i].text(0.97, 0.03, text, transform=axs[i].transAxes, fontsize=10,  va='bottom', ha='right', bbox=bbox)
        for idx in range(len(cv_obs_set)):
            obs_set_all_cv = cv_obs_set[idx]
            mean_obs = np.nanmean(obs_set_all_cv)
            axs[i].plot(mean_obs, actual_trends_obs_means[:, MONTH_IDX][idx], linestyle=linestyles[idx], marker=obs_markers[idx], markersize=8, color='k', label=obs_labels[idx])

for model_idx, set_of_preds_and_vals in enumerate(cv_preds_and_vals_):
    Y_pred = np.array(set_of_preds_and_vals[0])
    Y_test = np.array(set_of_preds_and_vals[1])

    for i in range(3):
        axs[i].scatter(Y_pred[:, i], Y_test[:, i], color=colors[model_idx], s=10, alpha=0.65, label=model_names_spliced[model_idx])
        axs[i].axline((0, 0), slope=1, color='k', alpha=0.5, linewidth=0.5)
        axs[i].set_xticks(ticks[i])
        axs[i].set_yticks(ticks[i])
        axs[i].set_ylim(ticks[i][0], ticks[i][-1])
        axs[i].set_xlim(ticks[i][0], ticks[i][-1])

all_preds_and_vals = np.concatenate(cv_preds_and_vals_, axis=1)
labels = ['A', 'B', 'C']

for i in range(3):
    axs[i].text(0.03, 0.97, labels[i], fontsize=14, transform=axs[i].transAxes, va='top', ha='left', fontweight='bold')
    Y_pred = all_preds_and_vals[0]
    Y_test = all_preds_and_vals[1]
    MSE = np.nanmean(np.square(Y_pred[:, i]-Y_test[:, i]), 0)
    MSE_str = str(np.around(MSE, 5))
    r = stats.pearsonr(Y_pred[:, i], Y_test[:, i])[0]
    r_str = str(np.around(r, 3))
    axs[i].set_title(titles[i]+'\nr: {:.2f}'.format(round(r, 2))+'      RMSE: {:.3f} K/dec'.format(round(np.sqrt(MSE), 3)), fontsize=12)
    axs[i].set_xlabel('Predicted Trend (K/dec)', fontsize=11)
    axs[0].set_ylabel('Actual Trend (K/dec)', fontsize=11)
    axs[i].grid(alpha=0.25)
    axs[i].tick_params(axis='both', which='major', labelsize=10)
    axs[i].tick_params(axis='both', which='minor', labelsize=10)

axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=11)
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/'+region+'_'+months[MONTH_IDX]+('_'+GEOMETRY_WEIGHTING if GEOMETRY_WEIGHTING else '')+'.png', dpi=300)
plt.show()

# Geometry-weighting sensitivity comparison
# To reproduce the sensitivity test referenced in the reviewer response:
# 1. Run this script once per geometry mode for the same month, e.g.:
#      qsub -F "april none" submit_cnn_7090.sh       (baseline)
#      qsub -F "april cos" submit_cnn_7090.sh
#      qsub -F "april sqrt_cos" submit_cnn_7090.sh    (optional)
#    Each run is saved with its own filename tag (none -> untagged, cos -> `_cos`,
#    sqrt_cos -> `_sqrt_cos`) so the runs don't overwrite each other.
# 2. Once the jobs you want to compare have finished, run the standalone
#    comparison script (no TensorFlow / training machinery required):
#      python compare_geometry.py --month april
