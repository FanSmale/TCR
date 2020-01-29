package datamodel;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import common.SimpleTools;

public class DataInfo_old {
	public static int userNumber = 24983;// 49290 //9439;//88238;//4696;
											// //1049511;//6040; 49290

	public static int itemNumber = 101;// 139738 // 139738; //66726;//3952;
										// 139738
	public static int rateNumber = 1810455; // 532274;// 4851475;
	public static int trNumber = 0;
	public static int teNumber = 0; // 132550;// 93100
	public static int teIndxRem = 2; // 0--9

	/********************** Feature Matrix ***********************************/
	public static short featureNumber = 30;
	public static double[][] uFeature = new double[userNumber][featureNumber];
	public static double[][] iFeature = new double[itemNumber][featureNumber];

	/******************** Training set *******************************/
	public static Triple[] data = new Triple[rateNumber];
	public static Triple[] GLdata = new Triple[rateNumber];

	/****
	 */
	public static int round = 500;
	public static double mean_rating = 0;

	public static double alpha = 0.0001;

	public static double lambda = 0.005;

	// public static int score_record = 0;

	public static String dataPath = new String("data/jester-data-1/jester-data-1.txt");
	// public static String testPath = new String("data/ml-100k/u1.test");
	static String split_Sign = new String("	");

	static int[] userCount;
	static int[] userPot;

	/**
	 * 
	 * @param paraFile
	 * @throws Exception
	 */
	public DataInfo_old(String paraDataPath) throws IOException {
		readData(paraDataPath);
		// readTestData(paraTestPath);
		setPot();
	}// of the first constructor

	/**
	 * 
	 * @param paraDataPath
	 * @throws IOException
	 */
	static void readData(String paraDataPath) throws IOException {
		File file = new File(paraDataPath);
		BufferedReader buffRead = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

		double sum = 0;
		int userIndex = 0;
		int index = 0;
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(split_Sign);

			int user = userIndex;// user id
			for (int i = 1; i < itemNumber; i++) {
				int item = i - 1;// item id
				double rating = Double.parseDouble(parts[i]);// rating

				if (rating != 99) {
					// data[index].i = user;
					// data[index].j = item;
					data[index] = new Triple(user, item, rating);
					if (index % 10 != teIndxRem) {
						sum += rating;// total rating
						trNumber++;
					} // Of if
					index++;
				} // Of if
			} // Of for i
			userIndex++;
		} // Of while

		teNumber = rateNumber - trNumber;
		System.out.println("index:" + index);
		mean_rating = sum / trNumber;// average rating
		for (int i = 0; i < DataInfo_old.rateNumber; i++) {
			double tmp = (Double) data[i].rating;// - mean_rating;
			data[i].rating = tmp;// ԭʼ����-ƽ����
		} // of for i
		buffRead.close();
	}

	/**
	 * 
	 */
	static void setPot() {
		userCount = new int[DataInfo_old.userNumber + 1];
		userPot = new int[DataInfo_old.userNumber + 1];

		for (int i = 0; i < DataInfo_old.rateNumber; i++) {
			// userCount[data[i].i]++;
			userCount[data[i].user]++;
		} // Of for i

		for (int i = 1; i <= DataInfo_old.userNumber; i++) {
			userPot[i] = userPot[i - 1] + userCount[i - 1];
		} // Of for i
	}// Of setPot

	/**
	 * 
	 * @param paraX
	 * @return
	 */
	double GLtransfer(double paraX) {
		double tempY = 0;
		double b = 1000;
		double v = 0.01;

		tempY = -1 / b * (Math.log(Math.pow(20 / (paraX + 10), v) - 1));

		return tempY;
	}// of GLtransfer

	/**
	 * 
	 * @param paraX
	 * @return
	 */
	public static double GLretransfer(double paraY) {
		double tempX = 0;
		double A = -10;
		double K = 10;
		double C = 1;
		double Q = 1;
		double B = 1000;
		double v = 0.01;

		tempX = A + (K - A) / (Math.pow((C + Q * Math.exp(-B * paraY)), 1 / v));

		return tempX;
	}// of GLtransfer

	/**
	 * 
	 */
	public void dataToGLData() {
		for (int i = 0; i < data.length; i++) {
			GLdata[i] = new Triple(data[i].user, data[i].item, GLtransfer(data[i].rating));

			System.out.println("oldRate: " + data[i].rating + " newRate: " + GLdata[i].rating);
		} // Of for i
	}// of dataToGLData

	/**
	 * 
	 * @param paraUser
	 * @param paraItem
	 * @return
	 */
	static Triple getDataInfo(int paraUser, int paraItem) {
		int left = userPot[paraUser];
		int right = userPot[paraUser + 1] - 1;

		while (left <= right) {
			int mid = (left + right) / 2;
			// if (data[mid].j > paraItem) {
			if (data[mid].item > paraItem) {
				right = mid - 1;
				// } else if (data[mid].j < paraItem) {
			} else if (data[mid].item < paraItem) {
				left = mid + 1;
			} else {
				return data[mid];
			} // of if
		} // of while
		return null;
	}// Of getDataInfo

	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			// Step 1. Initialize the train and test data based on group information
			DataInfo_old tempData = new DataInfo_old(dataPath);

			// Step3. Test
			//SimpleTools.printTriple(tempData.data);
			// Triple tempElement = getDataInfo(9, 6);
			// SimpleTool.printTriple(tempElement);

		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// of main
}// Of class DataInfo
