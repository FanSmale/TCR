package datamodel;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

import common.Common;
import datamodel.Triple;

/**
 * The basic data model. The data is organized in 2D of triples. Boolean means
 * that a boolean matrix indicates the training set. The purpose is to enable
 * incremental learning. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: January 20, 2020.<br>
 *       Last modified: February, 2020.
 * @version 1.0.
 */

public class RatingSystem2DBoolean {
	/**
	 * A sign to help reading the data file.
	 */
	public static final String SPLIT_SIGN_TAB = new String("	");

	/**
	 * A sign to help reading the data file.
	 */
	public static final String SPLIT_SIGN_COMMA = new String(",");
	
	/**
	 * The default missing rating.
	 */
	public static final int DEFAULT_MISSING_RATING = 99;
	

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
	 * @param paraCompress
	 *            Is the data in compress format?
	 ************************ 
	 */
	public RatingSystem2DBoolean(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			boolean paraCompress) {
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		numRatings = paraNumRatings;
		ratingLowerBound = paraRatingLowerBound;
		ratingUpperBound = paraRatingUpperBound;

		// Allocate space.
		data = new Triple[numUsers][];
		trainingIndicationMatrix = new boolean[numUsers][];

		itemPopularityArray = new int[numItems];
		itemRatingSumArray = new double[numItems];
		itemAverageRatingArray = new double[numItems];

		try {
			if (!paraCompress) {
				readData(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
			} else {
				readCompressedData(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
			} // Of
		} catch (Exception ee) {
			System.out.println("File " + paraFilename + " cannot be read! " + ee);
			System.exit(0);
		} // Of try
		
		for (int i = 0; i < numItems; i++) {
			// 0.0001 to avoid NaN due to unrated items.
			itemAverageRatingArray[i] = (itemRatingSumArray[i] + 0.0001)
					/ (itemPopularityArray[i] + 0.0001);
		} // Of for i
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
	private void readData(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings) throws IOException {
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(
				new InputStreamReader(new FileInputStream(file)));

		Triple[] tempTripleArrayForUser = new Triple[paraNumItems];
		int tempCurrentUserRatings = 0;

		int tempUserIndex = 0;
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(SPLIT_SIGN_TAB);

			// The first loop to read the current line for one user.
			tempCurrentUserRatings = 0;
			for (int i = 1; i < paraNumItems; i++) {
				int tempItemIndex = i - 1;// item id
				double tempRating = Double.parseDouble(parts[i]);// rating

				if (tempRating != DEFAULT_MISSING_RATING) {
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
	}// Of readData

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
	private void readCompressedData(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings) throws IOException {
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(
				new InputStreamReader(new FileInputStream(file)));

		Triple[] tempTripleArrayForUser = new Triple[paraNumItems];
		int tempUserNumRatings = 0;

		int tempLastUser = -1;
		int tempUser, tempItem;
		double tempRating;
		String[] tempParts;
		Triple tempNewTriple;
		for (int i = 0; i < paraNumRatings; i++) {
			tempParts = buffRead.readLine().split(SPLIT_SIGN_COMMA);

			tempUser = Integer.parseInt(tempParts[0]);
			tempItem = Integer.parseInt(tempParts[1]);
			tempRating = Double.parseDouble(tempParts[2]);

			tempNewTriple = new Triple(tempUser, tempItem, tempRating);

			if ((tempUser != tempLastUser) && (tempUser > 0)) {
				// Process the last user
				data[tempLastUser] = new Triple[tempUserNumRatings];
				for (int j = 0; j < tempUserNumRatings; j++) {
					data[tempLastUser][j] = tempTripleArrayForUser[j];
				} // Of for j
				trainingIndicationMatrix[tempLastUser] = new boolean[tempUserNumRatings];

				// Prepare for the next user
				tempUserNumRatings = 0;
			} // Of if

			tempTripleArrayForUser[tempUserNumRatings] = tempNewTriple;
			tempUserNumRatings++;

			itemPopularityArray[tempItem]++;
			itemRatingSumArray[tempItem] += tempRating;

			tempLastUser = tempUser;
		} // Of for i

		// Process the last user.
		data[tempLastUser] = new Triple[tempUserNumRatings];
		for (int j = 0; j < tempUserNumRatings; j++) {
			data[tempLastUser][j] = tempTripleArrayForUser[j];
		} // Of for j
		trainingIndicationMatrix[tempLastUser] = new boolean[tempUserNumRatings];

		buffRead.close();
	}// Of readCompressedData

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
		if ((paraTrainingItems == null) || (paraTrainingItems.length == 0)) {
			//System.out.println("Warning in RatingSystem2DBoolean(int, int[]):\r\n  user #"
			//		+ paraUser + " contains no training item.");
			return;
		} // Of if

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
	 ************************ 
	 */
	public double getMeanRating() {
		return meanRating;
	}// Of getMeanRating

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
	 * Get the user rating to the item.
	 * 
	 * @param paraUser
	 *            The index of the user.
	 * @param paraItem
	 *            The item.
	 ************************ 
	 */
	public double getUserItemRating(int paraUser, int paraItem) {
		for (int i = 0; i < data[paraUser].length; i++) {
			if (data[paraUser][i].item == paraItem) {
				return data[paraUser][i].rating;
			} else if (data[paraUser][i].item == paraItem) {
				break;
			}//Of if
		}//Of for i
		
		return DEFAULT_MISSING_RATING;
	}// Of getUserItemRating

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
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			boolean paraCompress) {
		RatingSystem2DBoolean tempRS = null;
		try {
			// Step 1. read the training and testing data
			tempRS = new RatingSystem2DBoolean(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound, paraCompress);
			System.out.println("" + tempRS.numUsers + " user, " + tempRS.numItems + " items, "
					+ tempRS.numRatings + " ratings. " + "\r\nAvearge ratings: "
					+ Arrays.toString(tempRS.itemAverageRatingArray));
		} catch (Exception e) {
			e.printStackTrace();
		} // of try

		for (int i = 0; i < 3; i++) {
			System.out.println(i);
			for (int j = 0; j < tempRS.data[i].length; j++) {
				System.out.println(tempRS.data[i][j]);
			} // Of for j
		} // Of for i

		int tempLength = 0;
		for (int i = 0; i < tempRS.numUsers; i++) {
			System.out.println(i);
			tempLength += tempRS.data[i].length;
		} // Of for i

		System.out.println("The read numRatings = " + tempLength);
	}// Of testReadingData

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testReadingData("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10,
				false);
		testReadingData("data/movielens943u1682m.txt", 943, 1682, 100000, 1, 5, true);
	}// Of main
}// Of class RatingSystem2DBoolean
