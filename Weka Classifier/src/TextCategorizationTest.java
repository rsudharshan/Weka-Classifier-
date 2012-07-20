import weka.core.*;
import weka.core.converters.*;
import weka.classifiers.Classifier;
import weka.classifiers.trees.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;

import java.io.*;
import java.lang.management.ClassLoadingMXBean;
import java.util.Iterator;

/**
 * Example class that converts HTML files stored in a directory structure into 
 * and ARFF file using the TextDirectoryLoader converter. It then applies the
 * StringToWordVector to the data and feeds a J48 classifier with it.
 *
 * @author Sudharshan
 */
public class TextCategorizationTest {
    static StringToWordVector filter = new StringToWordVector();
static Instances testFiltered=null;
  
public static Instances preprocessDir(String dirPath)
{
	TextDirectoryLoader loader = new TextDirectoryLoader();
	Instances dataRaw=null;
    try {
		loader.setDirectory(new File(dirPath));
	    dataRaw = loader.getDataSet();
	    loader.setOutputFilename(true);
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
     System.out.println(dataRaw);
    return dataRaw;
     
}
public static void trainModel(String modelPath,String dirPath,String algo)
{
		try
		{
		Classifier classifier=null;
	    if(algo.equals("naivebayes"))
	    {
	    	classifier=new NaiveBayes();
	    }
	    Instances dataRaw=preprocessDir(dirPath);
	    filter.setInputFormat(dataRaw);
	    Instances dataFiltered = Filter.useFilter(dataRaw, filter);
	    classifier.buildClassifier(dataFiltered);
	    ObjectOutputStream oos;
		
			oos = new ObjectOutputStream(new FileOutputStream(modelPath+".model"));
			oos.writeObject(classifier);
			oos.flush();
			oos.close();
		} catch (FileNotFoundException e) {
			
			e.printStackTrace();
		} catch (IOException e) {
			
			e.printStackTrace();
		} catch (Exception e) {
			
			e.printStackTrace();
		}
		
	}
public static Classifier getClassifier(String modelPath,String testDirPath)
{
	InputStream in;
	Classifier classifier=new NaiveBayes();
	try {
		    in = new FileInputStream(modelPath+".model");
			ObjectInputStream ois=new ObjectInputStream(in);
			classifier=(NaiveBayes)ois.readObject();
			ois.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return classifier;		
	}
  public static void main(String[] args) throws Exception {
    // convert the directory into a dataset
	  long start=System.currentTimeMillis();
    String dirPath="/home/developer/corpus/productr/300 reviews/train/";
    String testDirPath="/home/developer/corpus/productr/300 reviews/test/";
    String algo="naivebayes";
    String modelPath="/home/developer/workspace/Weka Classifier/";
    // train algo and output model

    trainModel(modelPath+algo, dirPath,algo);
   //loads the model 
    Classifier classifier = getClassifier(modelPath+algo, testDirPath);

    TextDirectoryLoader tester= new TextDirectoryLoader();
	tester.setDirectory(new File(testDirPath));
    Instances testSet=tester.getDataSet();
    filter.setInputFormat(testSet);
    testFiltered=Filter.useFilter(testSet, filter);
    System.out.println("\n\nClassifier model:\n\n" + classifier);
    int n=testFiltered.numInstances();
    int i=1;
    int ok=0;    
    while(i<n-1)
    {    	
    	Double d=classifier.classifyInstance(testSet.instance(i));
    	if(d.equals(testFiltered.instance(i).classValue()))
    	{
    		ok++;
    		System.out.println(testFiltered.instance(i).classValue());
    	}
		System.out.println(testFiltered.instance(i).classValue());
    	i++;
    	//System.out.println("Value: "+d+" "+s.classValue());    	
    }
    System.out.println("Correctly classified "+ok+" out of "+ n);
    long end=System.currentTimeMillis();
    System.out.println(end-start);
  }
}