# Haul-Truck-State-Classification-using-GPS-data
Classification of Haul truck Operational State in open-pit mining with GPS using LSTM


# Data Loading
We started by loading the telemetry and labels data from CSV files. Using Pandas made it pretty straightforward, especially with datetime parsing. This way, we could easily work with timestamps later.

# Preprocessing
Once the data was in, we needed to convert all relevant date columns into datetime objects. This involved converting the create_dt, start_time, and end_time columns. We then merged the telemetry data with the labels based on time intervals and the object name. Unfortunately, some rows ended up with missing operation states, so we had to drop those.

# Feature Engineering
This was an exciting part! We extracted several time-based features from the create_dt column:

Hour: To capture the time of day.
Day of the Week: To identify trends or patterns depending on the day.
Month: To see if there are seasonal effects.
We selected features for our LSTM model that included GPS speed and various accelerations, along with these time features. It felt like piecing together a puzzle.

# Normalization
Next, we normalized our feature values using MinMaxScaler. Scaling was crucial to ensure that the model trains effectively. We saved the scaler with Joblib to keep things consistent later on during validation.

# Creating Sequences
We created sequences of features with a length of 30 time steps. This way, each sequence of data would correlate with an operation state label. It was like training our model to recognize patterns over time.

# Train-Test Split
We split our data into training (80%) and testing (20%) sets. This was done using train_test_split from Scikit-learn, which ensured our model had enough data to learn from while also having a separate set to validate its performance.

# Model Building
We built our LSTM model using Keras. It was a sequential model with layers like LSTM, Dropout, and Dense. Compiling it with the Adam optimizer and sparse categorical cross-entropy loss gave us a good starting point.

# Model Training
Training the model was both nerve-wracking and exciting! We used EarlyStopping to prevent overfitting and ModelCheckpoint to save the best-performing model. Watching the validation metrics change over epochs was a mix of anticipation and hope.

# Model Evaluation
After training, we evaluated the model on the test set. It was a relief to see decent accuracy and loss metrics—it felt like our hard work was paying off!

# Validation
Finally, we moved to the validation dataset. We preprocessed and normalized it using the scaler we saved earlier. Creating sequences and making predictions with the trained model was the last step, and saving those predictions to a CSV file felt like tying a neat bow on our project.

# Tools Used
Pandas: Our go-to library for data manipulation. It made handling our datasets a breeze.
NumPy: For numerical operations; it’s like the backbone of our data handling.
Scikit-learn: Essential for preprocessing, scaling, and splitting our data.
TensorFlow/Keras: Where the magic of model building and training happened.
Joblib: A handy tool for saving our scaler to maintain consistency.
# Comparison Between Multiple Models
LSTM vs. Others:
LSTM is fantastic for sequences, allowing us to capture temporal dependencies effectively.
GRU: Similar to LSTM but often faster, which might help in quicker iterations.
CNN: While great for feature extraction, it might not excel in sequential data as much as LSTMs do.
Traditional ML Models: These can struggle with sequence data unless we do significant feature engineering.
# Struggles Faced
Data Merging: Aligning telemetry data with operational labels based on time was tricky. Timing is everything, right?
Missing Values: Dropping rows with missing operation states reduced our dataset size, which was a tough call.
Sequence Creation: Ensuring that the sequences aligned correctly with labels was challenging, especially with all the time steps involved.
Model Overfitting: Keeping an eye on training and validation loss was crucial to avoid overfitting, and implementing early stopping was our safety net.
