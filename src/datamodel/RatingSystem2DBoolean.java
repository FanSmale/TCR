package datamodel;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

import common.Common;
import datamodel.Triple;

/**
 * The basic data model. The data is organized in 2D of triples. Boolean means
 * that a boolean matrix indicates the training set. The purpose is to enable
 * incremental learning. Now only uncompressed data file is supported, that is,
 * missing value is indicated by 99. In the near future, the data organized by
 * triples should also be supported. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: January 20, 2020.<br>
 *       Last modified: January 30, 2020.
 * @version 1.0
 */

public class RatingSystem2DBoolean {
	/**
	 * A sign to help reading the data file.
	 */
	public static final String SPLIT_SIGN = new String("	");

	/**
	 * Number of users.
	 */
	protected int numUsers;

	/**
	 * Number of items.
	 */
	protected int numItems;

	/**
	 * Number of ratings.
	 */
	protected int numRatings;

	/**
	 * The whole data.
	 */
	protected Triple[][] data;

	/**
	 * The popularity of items. The ith user's popularity is data[i].length.
	 */
	public int[] itemPopularityArray;

	/**
	 * The sum of ratings of each item.
	 */
	double[] itemRatingSumArray;

	/**
	 * The average ratings of each item.
	 */
	double[] itemAverageRatingArray;

	/**
	 * Which elements belong to the training set.
	 */
	protected boolean[][] trainingIndicationMatrix;

	/**
	 * Mean rating calculated from the training sets.
	 */
	protected double meanRating;

	/**
	 * The lower bound of the rating value.
	 */
	protected double ratingLowerBound;

	/**
	 * The upper bound of the rating value.
	 */
	protected double ratingUpperBound;

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
	 * @param paraRatingLowerBound
	 *            The lower bound of ratings.
	 * @param paraRatingUpperBound
	 *            The upper bound of ratings.
	 ************************ 
	 */
	public RatingSystem2DBoolean(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound) {
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		numRatings = paraNumRatings;
		ratingLowerBound = paraRatingLowerBound;
		ratingUpperBound = paraRatingUpperBound;

		try {
			readData(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
		} catch (Exception ee) {
			System.out.println("File " + paraFilename + " cannot be read! " + ee);
			System.exit(0);
		} // Of try
	}// Of the first constructor

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 * @return The data in two-dimensional matrix of triples.
	 * @throws IOException
	 *             In case the file cannot be read.
	 ************************ 
	 */
	public Triple[][] readData(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings) throws IOException {
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(
				new InputStreamReader(new FileInputStream(file)));

		// Allocate space.
		data = new Triple[paraNumUsers][];
		trainingIndicationMatrix = new boolean[numUsers][];

		itemPopularityArray = new int[numItems];
		itemRatingSumArray = new double[numItems];
		itemAverageRatingArray = new double[numItems];

		Triple[] tempTripleArrayForUser = new Triple[paraNumItems];
		int tempCurrentUserRatings = 0;

		int tempUserIndex = 0;
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(SPLIT_SIGN);

			// The first loop to read the current line for one user.
			tempCurrentUserRatings = 0;
			for (int i = 1; i < paraNumItems; i++) {
				int tempItemIndex = i - 1;// item id
				double tempRating = Double.parseDouble(parts[i]);// rating

				if (tempRating != 99) {
					tempTripleArrayForUser[tempCurrentUserRatings] = new Triple(tempUserIndex,
							tempItemIndex, tempRating);
					tempCurrentUserRatings++;
					itemPopularityArray[tempItemIndex]++;
					itemRatingSumArray[tempItemIndex] += tempRating;
				} // Of if
			} // Of for i

			// The second loop to copy.
			data[tempUserIndex] = new Triple[tempCurrentUserRatings];
			trainingIndicationMatrix[tempUserIndex] = new boolean[tempCurrentUserRatings];
			for (int i = 0; i < tempCurrentUserRatings; i++) {
				data[tempUserIndex][i] = tempTripleArrayForUser[i];
			} // Of for i
			tempUserIndex++;
		} // Of while
		buffRead.close();

		for (int i = 0; i < numItems; i++) {
			// 0.01 to avoid NaN due to unrated items.
			itemAverageRatingArray[i] = itemRatingSumArray[i] / (itemPopularityArray[i] + 0.01);
		} // Of for i

		return data;
	}// Of readData

	/**
	 ************************ 
	 * Set the training part.
	 * 
	 * @param paraTrainingFraction
	 *            The fraction of the training set.
	 * @throws IOException
	 ************************ 
	 */
	public void initializeTraining(double paraTrainingFraction) {
		// Step 1. Read all data.
		int tempTotalTrainingSize = (int) (numRatings * paraTrainingFraction);
		int tempTotalTestingSize = numRatings - tempTotalTrainingSize;

		int tempTrainingSize = 0;
		int tempTestingSize = 0;
		double tempDouble;

		// Step 2. Handle each user.
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				tempDouble = Common.random.nextDouble();
				if (tempDouble <= paraTrainingFraction) {
					if (tempTrainingSize < tempTotalTrainingSize) {
						trainingIndicationMatrix[i][j] = true;
						tempTrainingSize++;
					} else {
						trainingIndicationMatrix[i][j] = false;
						tempTestingSize++;
					} // Of if
				} else {
					if (tempTestingSize < tempTotalTestingSize) {
						trainingIndicationMatrix[i][j] = false;
						tempTestingSize++;
					} else {
						trainingIndicationMatrix[i][j] = true;
						tempTrainingSize++;
					} // Of if
				} // Of if
			} // Of for j
		} // Of for i

		System.out.println("" + tempTrainingSize + " training instances.");
		System.out.println("" + tempTestingSize + " testing instances.");
	}// Of initializeTraining

	/**
	 ************************ 
	 * Set all data for training.
	 ************************ 
	 */
	public void setAllTraining() {
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				trainingIndicationMatrix[i][j] = true;
			} // Of for j
		} // Of for i
	}// Of setAllTraining

	/**
	 ************************ 
	 * Set all data of the user for training.
	 * 
	 * @param paraUser
	 *            The given user.
	 ************************ 
	 */
	public void setUserAllTraining(int paraUser) {
		for (int i = 0; i < trainingIndicationMatrix[paraUser].length; i++) {
			trainingIndicationMatrix[paraUser][i] = true;
		} // Of for i
	}// Of setUserAllTraining

	/**
	 ************************ 
	 * Set some data of the given user for training.
	 * 
	 * @param paraUser
	 *            The given user.
	 * @param paraTrainingIndices
	 *            The item indices for the given user as training.
	 ************************ 
	 */
	public void setUserTraining(int paraUser, int[] paraTrainingItems) {
		int tempItemIndex = 0;
		int i;
		for (i = 0; i < trainingIndicationMatrix[paraUser].length; i++) {
			if (data[paraUser][i].item == paraTrainingItems[tempItemIndex]) {
				trainingIndicationMatrix[paraUser][i] = true;
				tempItemIndex++;
				if (tempItemIndex == paraTrainingItems.length) {
					break;
				} // Of if
			} else {
				trainingIndicationMatrix[paraUser][i] = false;
			} // Of if
		} // Of for j

		// The remaining parts are all testing.
		// Attention: i should not be re-initialized!
		for (; i < trainingIndicationMatrix[paraUser].length; i++) {
			trainingIndicationMatrix[paraUser][i] = false;
		} // Of for i
	}// Of setUserTraining
	
	/**
	 ************************ 
	 * Getter.
	 ************************ 
	 */
	public int getNumUsers() {
		return numUsers;
	}// Of getNumUsers

	/**
	 ************************ 
	 * Getter.
	 ************************ 
	 */
	public int getNumItems() {
		return numItems;
	}// Of getNumItems

	/**
	 ************************ 
	 * Getter.
	 ************************ 
	 */
	public int getNumRatings() {
		return numRatings;
	}// Of getNumRatings

	/**
	 ************************ 
	 * Getter. Get the number of ratings of the user.
	 ************************ 
	 */
	public int getUserNumRatings(int paraUser) {
		return data[paraUser].length;
	}// Of getUserNumRatings

	/**
	 ************************ 
	 * Getter.
	 ************************ 
	 */
	public double getRatingLowerBound() {
		return ratingLowerBound;
	}// Of getRatingsLowerBound

	/**
	 ************************ 
	 * Getter.
	 ************************ 
	 */
	public double getRatingUpperBound() {
		return ratingUpperBound;
	}// Of getRatingsUpperBound

	/**
	 ************************ 
	 * Getter.
	 * 
	 * @param paraUser
	 *            The index of the user.
	 * @param paraIndex
	 *            The jth item rated by the user, instead of the jth item of the
	 *            dataset.
	 ************************ 
	 */
	public boolean getTrainIndication(int paraUser, int paraIndex) {
		return trainingIndicationMatrix[paraUser][paraIndex];
	}// Of getTrainIndication

	/**
	 ************************ 
	 * Setter.
	 * 
	 * @param paraUser
	 *            The index of the user.
	 * @param paraIndex
	 *            The jth item rated by the user, instead of the jth item of the
	 *            dataset.
	 ************************ 
	 */
	public void setTrainIndication(int paraUser, int paraIndex, boolean paraValue) {
		trainingIndicationMatrix[paraUser][paraIndex] = paraValue;
	}// Of setTrainIndication

	/**
	 ************************ 
	 * Getter.
	 * 
	 * @param paraUser
	 *            The index of the user.
	 * @param paraIndex
	 *            The jth item rated by the user, instead of the jth item of the
	 *            dataset.
	 ************************ 
	 */
	public Triple getTriple(int paraUser, int paraIndex) {
		return data[paraUser][paraIndex];
	}// Of getTriple

	/**
	 ************************ 
	 * Getter.
	 ************************ 
	 */
	public int getItemPopularity(int paraItem) {
		return itemPopularityArray[paraItem];
	}// Of getItemPopularity

	/**
	 *********************************** 
	 * Show me.
	 *********************************** 
	 */
	public String toString() {
		String resultString = "I am 2D rating system.\r\n";
		resultString += "I have " + numUsers + " users, " + numItems + " items, and " + numRatings
				+ " ratings.";
		return resultString;
	}// Of toString

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
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				if (trainingIndicationMatrix[i][j]) {
					tempRatingSum += data[i][j].rating;
					tempTrainingSize++;
				} // Of if
			} // Of for j
		} // Of for i
		meanRating = tempRatingSum / tempTrainingSize;

		// Step 2. Update the ratings in the training set.
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				if (trainingIndicationMatrix[i][j]) {
					data[i][j].rating -= meanRating;
				} // Of if
			} // Of for j
		} // Of for i

		// Step 3. Also update the bounds.
		ratingLowerBound -= meanRating;
		ratingUpperBound -= meanRating;
	}// Of adjustUsingMeanRating

	/**
	 ************************ 
	 * Training and testing using the same data.
	 ************************ 
	 */
	public static void testReadingData(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound) {
		try {
			// Step 1. read the training and testing data
			RatingSystem2DBoolean tempMF = new RatingSystem2DBoolean(paraFilename, paraNumUsers,
					paraNumItems, paraNumRatings, paraRatingLowerBound, paraRatingUpperBound);
			System.out.println("" + tempMF.numUsers + " user, " + tempMF.numItems + " items, "
					+ tempMF.numRatings + " ratings. " + "\r\nAvearge ratings: "
					+ Arrays.toString(tempMF.itemAverageRatingArray));
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testReadingData

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testReadingData("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10);
	}// Of main
}// Of class RatingSystem2DBoolean
