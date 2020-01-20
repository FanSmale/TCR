package algorithm;

import java.io.*;
import java.util.Random;
import datamodel.Triple;

/*
 * @(#)MatrixFactorization2D.java
 
 * Project: Matrix factorization for recommender systems.
 * The data is organized in 2D to enable incremental learning. 
 * Author: Fan Min, Yuan-Yuan Xu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.
 * Created: December 3, 2019.
 * Last modified: Jan 19, 2020.
 */

public class MatrixFactorization2D {
	/**
	 * A sign to help reading the data file.
	 */
	public static final String SPLIT_SIGN = new String("	");

	/**
	 * Used to generate random numbers.
	 */
	Random rand = new Random();

	/**
	 * Number of users.
	 */
	int numUsers;

	/**
	 * Number of items.
	 */
	int numItems;

	/**
	 * Number of ratings.
	 */
	int numRatings;

	/**
	 * Training data.
	 */
	Triple[][] trainingData;

	/**
	 * Testing data.
	 */
	Triple[][] testingData;

	/**
	 * Mean rating calculated from the training sets.
	 */
	double meanRating;

	/**
	 * A parameter for controlling learning regular.
	 */
	double alpha;

	/**
	 * A parameter for controlling the learning speed.
	 */
	double lambda;

	/**
	 * The low rank of the small matrices.
	 */
	int rank;

	/**
	 * The user matrix U.
	 */
	double[][] userSubspace;

	/**
	 * The item matrix V.
	 */
	double[][] itemSubspace;

	/**
	 * The lower bound of the rating value.
	 */
	double ratingLowerBound;

	/**
	 * The upper bound of the rating value.
	 */
	double ratingUpperBound;

	/**
	 * Regular scheme.
	 */
	int regularScheme;

	/**
	 * No regular scheme.
	 */
	public static final int NO_REGULAR = 0;

	/**
	 * No regular scheme.
	 */
	public static final int PQ_REGULAR = 1;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraFilename
	 *            The data filename.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 ************************ 
	 */
	public MatrixFactorization2D(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, double paraTrainingFraction) {
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		numRatings = paraNumRatings;
		ratingLowerBound = paraRatingLowerBound;
		ratingUpperBound = paraRatingUpperBound;

		// data = new Triple[numRatings];
		try {
			readData(paraFilename, paraTrainingFraction);
		} catch (Exception ee) {
			System.out.println("File " + paraFilename + " cannot be read! " + ee);
			System.exit(0);
		} // Of try

		initialize();
	}// Of the first constructor

	/**
	 ************************ 
	 * Set parameters.
	 * 
	 * @param paraRank
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void setParameters(int paraRank, double paraAlpha, double paraLambda, int paraRegularScheme) {
		rank = paraRank;
		alpha = paraAlpha;
		lambda = paraLambda;
		regularScheme = paraRegularScheme;
	}// Of setParameters

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public static Triple[] readData(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings)
			throws IOException {
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

		// Allocate space.
		Triple[] resultData = new Triple[paraNumRatings];

		int tempUserIndex = 0;
		int tempIndex = 0;
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(SPLIT_SIGN);

			for (int i = 1; i < paraNumItems; i++) {
				int tempItemIndex = i - 1;// item id
				double tempRating = Double.parseDouble(parts[i]);// rating

				if (tempRating != 99) {
					resultData[tempIndex] = new Triple(tempUserIndex, tempItemIndex, tempRating);
					tempIndex++;
				} // Of if
			} // Of for i
			tempUserIndex++;
		} // Of while
		buffRead.close();

		return resultData;
	}// Of readData

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraTrainingFraction
	 *            The fraction of the training set.
	 * @throws IOException
	 ************************ 
	 */
	public void readData(String paraFilename, double paraTrainingFraction) throws IOException {
		// Step 1. Read all data.
		Triple[] tempAllData = readData(paraFilename, numUsers, numItems, numRatings);

		int tempNumTrainingRatings = (int) (numRatings * paraTrainingFraction);
		int tempNumTestingRatings = numRatings - tempNumTrainingRatings;
		System.out.println("numTrainingRatings = " + tempNumTrainingRatings);
		System.out.println("numTestingRatings = " + tempNumTestingRatings);

		trainingData = new Triple[numUsers][];
		testingData = new Triple[numUsers][];

		// Step 2. Calculate the number of ratings for each user.
		int[] tempRatingForUserCounts = new int[numUsers];
		for (int i = 0; i < tempAllData.length; i++) {
			tempRatingForUserCounts[tempAllData[i].user]++;
		} // Of for i

		int tempTrainingSize = 0;
		int tempTestingSize = 0;
		double tempDouble;
		int tempRatingSum = 0;

		// Step 3. Handle each user.
		for (int i = 0; i < numUsers; i++) {
			int tempNumTraining = 0;
			int tempNumTesting = 0;
			boolean[] tempIsTrainArray = new boolean[tempRatingForUserCounts[i]];

			// Step 3.1 First scan to determine which ones belong to training
			for (int j = 0; j < tempRatingForUserCounts[i]; j++) {
				tempDouble = rand.nextDouble();
				if (tempDouble <= paraTrainingFraction) {
					if (tempTrainingSize < tempNumTrainingRatings) {
						tempIsTrainArray[j] = true;
						tempNumTraining++;
					} else {
						tempIsTrainArray[j] = false;
						tempNumTesting++;
					} // Of if
				} else {
					if (tempTestingSize < tempNumTestingRatings) {
						tempIsTrainArray[j] = false;
						tempNumTesting++;
					} else {
						tempIsTrainArray[j] = true;
						tempNumTraining++;
					} // Of if
				} // Of if
			} // Of for j

			// Step 3.2 Allocate space.
			trainingData[i] = new Triple[tempNumTraining];
			testingData[i] = new Triple[tempNumTesting];

			// Step 3.3 Second scan for copy.
			int tempTrainingCount = 0;
			int tempTestingCount = 0;
			for (int j = 0; j < tempRatingForUserCounts[i]; j++) {
				if (tempIsTrainArray[j]) {
					trainingData[i][tempTrainingCount] = tempAllData[tempRatingSum + j];
					tempTrainingCount++;
				} else {
					testingData[i][tempTestingCount] = tempAllData[tempRatingSum + j];
					tempTestingCount++;
				} // Of if
			} // Of for j

			// Update the total number of ratings for processed users.
			tempRatingSum += tempRatingForUserCounts[i];
		} // Of for i

		// Step 3. Compute meanRating
		adjustUsingMeanRating();
	}// Of readData

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void readTrainingData(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings)
			throws IOException {
		// Step 1. Read all data.
		Triple[] tempAllData = readData(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);

		trainingData = new Triple[numUsers][];

		// Step 2. Calculate the number of ratings for each user.
		int[] tempRatingForUserCounts = new int[numUsers];
		for (int i = 0; i < tempAllData.length; i++) {
			tempRatingForUserCounts[tempAllData[i].user]++;
		} // Of for i

		int tempRatingSum = 0;

		// Step 3. Handle each user.
		for (int i = 0; i < numUsers; i++) {
			// Step 3.1 Allocate space.
			trainingData[i] = new Triple[tempRatingForUserCounts[i]];

			// Step 3.2 Copy.
			for (int j = 0; j < tempRatingForUserCounts[i]; j++) {
				trainingData[i][j] = tempAllData[tempRatingSum + j];
			} // Of for j

			// Update the total number of ratings for processed users.
			tempRatingSum += tempRatingForUserCounts[i];
		} // Of for i

		// Step 3. Compute meanRating
		adjustUsingMeanRating();
	}// Of readTrainingData

	/**
	 ************************ 
	 * Adjust the training data with the mean rating. The ratings are subtracted
	 * with the mean rating. So do the rating bounds.
	 ************************ 
	 */
	public void adjustUsingMeanRating() {
		// Step 1. Calculate the mean rating.
		double tempRatingSum = 0;
		int tempTrainingSize = 0;
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingData[i].length; j++) {
				tempRatingSum += trainingData[i][j].rating;
			} // Of for j
			tempTrainingSize += trainingData[i].length;
		} // Of for i
		meanRating = tempRatingSum / tempTrainingSize;

		// Step 2. Update the ratings in the training set.
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingData[i].length; j++) {
				trainingData[i][j].rating -= meanRating;
			} // Of for j
		} // Of for i

		// Step 3. Also update the bounds.
		ratingLowerBound -= meanRating;
		ratingUpperBound -= meanRating;
	}// Of adjustUsingMeanRating

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void readTestingData(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings)
			throws IOException {
		// Step 1. Read all data.
		Triple[] tempAllData = readData(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);

		testingData = new Triple[numUsers][];

		// Step 2. Calculate the number of ratings for each user.
		int[] tempRatingForUserCounts = new int[numUsers];
		for (int i = 0; i < tempAllData.length; i++) {
			tempRatingForUserCounts[tempAllData[i].user]++;
		} // Of for i

		int tempRatingSum = 0;

		// Step 3. Handle each user.
		for (int i = 0; i < numUsers; i++) {
			// Step 3.1 Allocate space.
			testingData[i] = new Triple[tempRatingForUserCounts[i]];

			// Step 3.2 Copy.
			for (int j = 0; j < tempRatingForUserCounts[i]; j++) {
				testingData[i][j] = tempAllData[tempRatingSum + j];
			} // Of for j

			// Update the total number of ratings for processed users.
			tempRatingSum += tempRatingForUserCounts[i];
		} // Of for i
	}// Of readTestingData

	/**
	 ************************ 
	 * Initialize some variables.
	 ************************ 
	 */
	void initialize() {
		rank = 5;
		alpha = 0.0001;
		lambda = 0.005;
		regularScheme = 0;
	}// Of initialize

	/**
	 ************************ 
	 * Initialize subspaces. Each value is in [-paraRange, +paraRange].
	 * 
	 * @paraRange The range of the initial values.
	 ************************ 
	 */
	void initializeSubspaces(double paraRange) {
		userSubspace = new double[numUsers][rank];

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < rank; j++) {
				userSubspace[i][j] = (rand.nextDouble() - 0.5) * 2 * paraRange;
			} // of for j
		} // Of for i

		// SimpleTool.printMatrix(DataInfo.userFeature);
		itemSubspace = new double[numItems][rank];
		for (int i = 0; i < numItems; i++) {
			for (int j = 0; j < rank; j++) {
				itemSubspace[i][j] = 1 * (rand.nextDouble() - 0.5) * 2 * paraRange;
			} // Of for j
		} // Of for i
	}// of initializeSubspaces

	/**
	 ************************ 
	 * Predict the rating of the user to the item
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	 */
	public double predict(int paraUser, int paraItem) {
		// System.out.println("Predict in the superclass");
		double resultValue = 0;
		for (int i = 0; i < rank; i++) {
			// User����������Item�������ĳ˻�. ���ִ洢��ʽ�ǳ�����
			resultValue += userSubspace[paraUser][i] * itemSubspace[paraItem][i];
		} // of for i
		return resultValue;
	}// Of predict

	/**
	 ************************ 
	 * Train.
	 * 
	 * @param paraRounds
	 *            The number of rounds.
	 ************************ 
	 */
	public void train(int paraRounds) {
		for (int i = 0; i < paraRounds; i++) {
			update();
			if (i % 50 == 0) {
				// Show the process
				System.out.println("Round " + i);
				System.out.println("MAE: " + mae());
			} // Of if
		} // Of for i
	}// Of train

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void update() {
		switch (regularScheme) {
		case NO_REGULAR:
			updateNoRegular();
			break;
		case PQ_REGULAR:
			updatePQRegular();
			break;
		default:
			System.out.println("Unsupported regular scheme: " + regularScheme);
			System.exit(0);
		}// Of switch
	}// Of update

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updateNoRegular() {
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingData[i].length; j++) {
				int tempUserId = (int) trainingData[i][j].user;
				int tempItemId = (int) trainingData[i][j].item;
				double tempRate = (double) trainingData[i][j].rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k];

					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of updateNoRegular

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updatePQRegular() {
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingData[i].length; j++) {
				int tempUserId = (int) trainingData[i][j].user;
				int tempItemId = (int) trainingData[i][j].item;
				double tempRate = (double) trainingData[i][j].rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k] - lambda * userSubspace[tempUserId][k];

					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k] - lambda * itemSubspace[tempItemId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of updatePQRegular

	/**
	 ************************ 
	 * Compute the RSME.
	 * 
	 * @return RSME of the current factorization.
	 ************************ 
	 */
	public double rsme() {
		double resultRsme = 0;
		int tempTestCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < testingData[i].length; j++) {
				int tempUserIndex = testingData[i][j].user;
				int tempItemIndex = testingData[i][j].item;
				double tempRate = testingData[i][j].rating;

				double tempPrediction = predict(tempUserIndex, tempItemIndex);// +
																				// DataInfo.mean_rating;

				if (tempPrediction < ratingLowerBound) {
					tempPrediction = ratingLowerBound;
				} else if (tempPrediction > ratingUpperBound) {
					tempPrediction = ratingUpperBound;
				} // Of if

				double tempError = tempRate - tempPrediction;
				resultRsme += tempError * tempError;
				tempTestCount++;
			} // Of for j
		} // Of for i

		return Math.sqrt(resultRsme / tempTestCount);
	}// Of rsme

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public double mae() {
		double resultMae = 0;
		int tempTestCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < testingData[i].length; j++) {
				int tempUserIndex = testingData[i][j].user;
				int tempItemIndex = testingData[i][j].item;
				double tempRate = testingData[i][j].rating;

				double tempPrediction = predict(tempUserIndex, tempItemIndex);

				if (tempPrediction < ratingLowerBound) {
					tempPrediction = ratingLowerBound;
				} // Of if
				if (tempPrediction > ratingUpperBound) {
					tempPrediction = ratingUpperBound;
				} // of if

				double tempError = tempRate - tempPrediction;

				resultMae += Math.abs(tempError);
				// System.out.println("resultMae: " + resultMae);
				tempTestCount++;
			} // Of for j
		} // Of for i

		return (resultMae / tempTestCount);
	}// Of mae

	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testTrainingTesting(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			MatrixFactorization2D tempMF = new MatrixFactorization2D(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound, 0.9);

			tempMF.setParameters(10, 0.0001, 0.005, PQ_REGULAR);
			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			tempMF.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempMF.train(paraRounds);

			double tempMAE = tempMF.mae();
			double tempRSME = tempMF.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testTrainingTesting

	/**
	 ************************ 
	 * The training and testing sets are the same.
	 ************************ 
	 */
	public static void testSameTrainingTesting(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			MatrixFactorization2D tempMF = new MatrixFactorization2D(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound, 0.9);

			// Test only!
			 tempMF.readTrainingData(paraFilename, paraNumUsers, paraNumItems,
			 paraNumRatings);
			 tempMF.readTestingData(paraFilename, paraNumUsers, paraNumItems,
			 paraNumRatings);

			tempMF.setParameters(10, 0.0001, 0.005, PQ_REGULAR);
			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			tempMF.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempMF.train(paraRounds);

			double tempMAE = tempMF.mae();
			double tempRSME = tempMF.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testTrainingTesting	
	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testTrainingTesting("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, 300);
		//testSameTrainingTesting("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, 500);
	}// Of main
}// Of class MatrixFactorization2D
