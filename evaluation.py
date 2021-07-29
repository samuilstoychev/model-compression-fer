"""Auxiliary function for evaluating model performance"""
import tempfile 
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
import os

def get_model_size(model):
    # Create a temporary file to store the model
    _, file = tempfile.mkstemp('.h5')
    # Save the Keras model in that file
    tf.keras.models.save_model(model, file, include_optimizer=False)
    # Zip the model (zipping is needed to see the benefits of compression)
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    # Return the zip file size 
    return os.path.getsize(zipped_file)

def get_results(model, test_generator): 
    results = [] 

    for batch_number in range(int(np.ceil(test_generator.n / test_generator.batch_size))): 
        data = test_generator[batch_number][0]
        labels = test_generator[batch_number][1]
        predictions = model.predict(data)
        filenames = [test_generator.filenames[batch_number * test_generator.batch_size + i] for i in range(len(data))]
        results.append((filenames, data, labels, predictions)) 
    
    results_flattened = [] 
    
    for result in results: 
        filenames, data, labels, predictions = result 
        for i in range(len(data)): 
            results_flattened.append((filenames[i], data[i], labels[i], predictions[i]))
            
    table_data = []

    for result in results_flattened: 
        filename, data, label, prediction = result 

        gender = filename[-5]
        prediction_true = (np.argmax(label) == np.argmax(prediction))

        table_data.append([gender, np.argmax(label), np.argmax(prediction), prediction_true])
    
    return pd.DataFrame(table_data, columns=["Gender", "Ground Truth", "Prediction", "Prediction Correct"])

def get_results_rafdb(model, test_generator): 
    results = [] 

    for batch_number in range(int(np.ceil(test_generator.n / test_generator.batch_size))): 
        data = test_generator[batch_number][0]
        labels = test_generator[batch_number][1]
        predictions = model.predict(data)
        filenames = [test_generator.filenames[batch_number * test_generator.batch_size + i] for i in range(len(data))]
        results.append((filenames, data, labels, predictions)) 
    
    results_flattened = [] 
    
    for result in results: 
        filenames, data, labels, predictions = result 
        for i in range(len(data)): 
            results_flattened.append((filenames[i], data[i], labels[i], predictions[i]))
            
    table_data = []

    for result in results_flattened: 
        filename, data, label, prediction = result 
        try: 
            _, gender, race, age = ((filename[:-4]).split("/"))[1].split("_")
        except Exception as e: 
            print("Failed for", filename)
            print(e)
            raise
        prediction_true = (np.argmax(label) == np.argmax(prediction))

        table_data.append([gender, race, age, np.argmax(label), np.argmax(prediction), prediction_true])
    
    return pd.DataFrame(table_data, columns=["Gender", "Race", "Age", "Ground Truth", "Prediction", "Prediction Correct"])

def get_specific_accuracy(results, gender, ground_truth): 
    query_string = "Gender == '%s' and `Ground Truth` == %d" % (gender, ground_truth) 
    # The resulting DataFrame will only include the results for subjects 
    # with gender=<gender> and emotion=<ground_truth>
    subset = results.query(query_string)
    return sum(subset["Prediction Correct"]) / len(subset)

def get_metrics(model, test_generator): 
    """Evaluate `model` on the data from `test_generator`."""
    metrics = {}
    metrics['size'] = get_model_size(model)
    df = get_results(model, test_generator)
    metrics['acc'] = sum(df["Prediction Correct"]) / len(df["Prediction Correct"])
    f_df = df[df["Gender"] == "f"]
    m_df = df[df["Gender"] == "m"]
    metrics['f_acc'] = sum(f_df["Prediction Correct"]) / len(f_df["Prediction Correct"])
    metrics['m_acc'] = sum(m_df["Prediction Correct"]) / len(m_df["Prediction Correct"])
    
    f_accs = [] 
    for i in range(8): 
        acc = get_specific_accuracy(df, "f", i)
        f_accs.append(acc)
    metrics['f_acc_breakdown'] = f_accs
    metrics['f_acc_balanced'] = sum(f_accs)/len(f_accs)
    
    m_accs = [] 
    for i in range(8): 
        acc = get_specific_accuracy(df, "m", i)
        m_accs.append(acc)
    metrics['m_acc_breakdown'] = m_accs
    metrics['m_acc_balanced'] = sum(m_accs)/len(m_accs)
    
    return metrics

def get_metrics_rafdb(model, test_generator):
    """Evaluate `model` on the data from `test_generator`."""
    metrics = {}
    metrics['size'] = get_model_size(model)
    df = get_results_rafdb(model, test_generator)

    metrics['acc'] = sum(df["Prediction Correct"]) / len(df["Prediction Correct"])
    f_df = df[df["Gender"] == "1"]
    m_df = df[df["Gender"] == "0"]
    metrics['f_acc'] = sum(f_df["Prediction Correct"]) / len(f_df["Prediction Correct"])
    metrics['m_acc'] = sum(m_df["Prediction Correct"]) / len(m_df["Prediction Correct"])

    caucasian_df = df[df["Race"] == "0"]
    afamerican_df = df[df["Race"] == "1"]
    asian_df = df[df["Race"] == "2"]
    metrics['caucasian_acc'] = sum(caucasian_df["Prediction Correct"]) / len(caucasian_df["Prediction Correct"])
    metrics['afamerican_acc'] = sum(afamerican_df["Prediction Correct"]) / len(afamerican_df["Prediction Correct"])
    metrics['asian_acc'] = sum(asian_df["Prediction Correct"]) / len(asian_df["Prediction Correct"])

    age0_df = df[df["Age"] == "0"]
    age1_df = df[df["Age"] == "1"]
    age2_df = df[df["Age"] == "2"]
    age3_df = df[df["Age"] == "3"]
    age4_df = df[df["Age"] == "4"]
    metrics['age0_acc'] = sum(age0_df["Prediction Correct"]) / len(age0_df["Prediction Correct"])
    metrics['age1_acc'] = sum(age1_df["Prediction Correct"]) / len(age1_df["Prediction Correct"])
    metrics['age2_acc'] = sum(age2_df["Prediction Correct"]) / len(age2_df["Prediction Correct"])
    metrics['age3_acc'] = sum(age3_df["Prediction Correct"]) / len(age3_df["Prediction Correct"])
    metrics['age4_acc'] = sum(age4_df["Prediction Correct"]) / len(age4_df["Prediction Correct"])
    
    return metrics
# ===========================================================================
# =========== AUXILIARY FUNCTIONS FOR EVALUATING QUANTISED MODELS ===========
# ===========================================================================

def get_zipped_model_size(file):

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)

def get_results_tflite(interpreter, test_generator, dataset="ckplus"): 
    """Get the prediction results from a TFLite file (representing a compressed model)"""
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    # Run predictions on ever y image in the "test" dataset.
    predicted_emotions = []
    
    test_data_flat = [] 
    test_labels_flat = [] 
    for batch_number in range(int(np.ceil(test_generator.n / test_generator.batch_size))):
        for image in test_generator[batch_number][0]:
            test_data_flat.append(image)
        for label in test_generator[batch_number][1]: 
            test_labels_flat.append(label)
    
    for i, test_image in enumerate(test_data_flat):
        if i % 100 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        emotion = np.argmax(output()[0])
        predicted_emotions.append(emotion)

    # Compare prediction results with ground truth labels to calculate accuracy.
    gt_emotions = [np.argmax(x) for x in test_labels_flat]
    if dataset == "ckplus": 
        genders = [x[-5] for x in test_generator.filenames]

        return pd.DataFrame({"Gender": genders, 
                         "Ground Truth": gt_emotions, 
                         "Prediction": predicted_emotions, 
                         "Prediction Correct": np.array(predicted_emotions) == np.array(gt_emotions)})
    else: 
        genders = [((x[:-4]).split("/"))[1].split("_")[1] for x in test_generator.filenames]
        races = [((x[:-4]).split("/"))[1].split("_")[2] for x in test_generator.filenames]
        ages = [((x[:-4]).split("/"))[1].split("_")[3] for x in test_generator.filenames]

        return pd.DataFrame({"Gender": genders, 
                         "Race": races, 
                         "Age": ages, 
                         "Ground Truth": gt_emotions, 
                         "Prediction": predicted_emotions, 
                         "Prediction Correct": np.array(predicted_emotions) == np.array(gt_emotions)})


def get_metrics_quantised(model, test_generator, dataset="ckplus"): 

    # Save quantised model as a file 
    _, quantized_tflite_file = tempfile.mkstemp('.tflite')
    with open(quantized_tflite_file, 'wb') as f:
        f.write(model)

    metrics = {}
    metrics['size'] = get_zipped_model_size(quantized_tflite_file)

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    
    df = get_results_tflite(interpreter, test_generator, dataset)
    metrics['acc'] = sum(df["Prediction Correct"]) / len(df["Prediction Correct"])

    if dataset == "ckplus": 
        f_df = df[df["Gender"] == "f"]
        m_df = df[df["Gender"] == "m"]
        metrics['f_acc'] = sum(f_df["Prediction Correct"]) / len(f_df["Prediction Correct"])
        metrics['m_acc'] = sum(m_df["Prediction Correct"]) / len(m_df["Prediction Correct"])
    else: 
        f_df = df[df["Gender"] == "1"]
        m_df = df[df["Gender"] == "0"]
        metrics['f_acc'] = sum(f_df["Prediction Correct"]) / len(f_df["Prediction Correct"])
        metrics['m_acc'] = sum(m_df["Prediction Correct"]) / len(m_df["Prediction Correct"])

        caucasian_df = df[df["Race"] == "0"]
        afamerican_df = df[df["Race"] == "1"]
        asian_df = df[df["Race"] == "2"]
        metrics['caucasian_acc'] = sum(caucasian_df["Prediction Correct"]) / len(caucasian_df["Prediction Correct"])
        metrics['afamerican_acc'] = sum(afamerican_df["Prediction Correct"]) / len(afamerican_df["Prediction Correct"])
        metrics['asian_acc'] = sum(asian_df["Prediction Correct"]) / len(asian_df["Prediction Correct"])

        age0_df = df[df["Age"] == "0"]
        age1_df = df[df["Age"] == "1"]
        age2_df = df[df["Age"] == "2"]
        age3_df = df[df["Age"] == "3"]
        age4_df = df[df["Age"] == "4"]
        metrics['age0_acc'] = sum(age0_df["Prediction Correct"]) / len(age0_df["Prediction Correct"])
        metrics['age1_acc'] = sum(age1_df["Prediction Correct"]) / len(age1_df["Prediction Correct"])
        metrics['age2_acc'] = sum(age2_df["Prediction Correct"]) / len(age2_df["Prediction Correct"])
        metrics['age3_acc'] = sum(age3_df["Prediction Correct"]) / len(age3_df["Prediction Correct"])
        metrics['age4_acc'] = sum(age4_df["Prediction Correct"]) / len(age4_df["Prediction Correct"])
            
    return metrics