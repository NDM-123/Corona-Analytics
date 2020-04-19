package com.example.coronaanalytics;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;


public class FaceDataBase {
    static HashMap<String, Mat> names = new HashMap<>();
    static ArrayList<Mat> faces = new ArrayList<>();
    public static final String jdbcUrl = "";//define database
    public static final String jdbcUser = "";//define username mysql
    public static final String jdbcUserPassword = "";//define username password

    public FaceDataBase() {
        names = null;
        faces = null;
    }

    public static void connectDb() {
        try {
            Connection connection =
                    DriverManager.getConnection(jdbcUrl, jdbcUser, jdbcUserPassword);
            Statement statement = connection.createStatement();
            String allFaces = "SELECT * FROM faces;";
            ResultSet resultSet = statement.executeQuery(allFaces);
            while (resultSet.next()) {

                // Deserialization

                    // Method for deserialization of object
                Mat ma  = (Mat)resultSet.getObject("face");

                names.put(resultSet.getString("Name"), ma);
                faces.add(ma);

                }
        connection.close();


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
//    public static void uploadDb() {       //maybe in future use to update data base
//        try {
//            Connection connection =
//                    DriverManager.getConnection(jdbcUrl, jdbcUser, jdbcUserPassword);
//            Statement statement = connection.createStatement();
//            String allFaces = "Insert into faces(Name,face) values(?,?);";
//            connection.prepareStatement(allFaces);
//            String name="";
//            int i=0;
//            for(Mat face:faces) {
//
//                for ( String key : names.keySet() )
//                {
//                    if ( names.get( key ).equals( face ) )
//                    {
//                        name=key;
//                        break;
//                    }
//                }
//                // serialization
//                statement.set
//                statement.setObject(name,face);
//                // Method for deserialization of object
//                Mat ma  = names.updateObject(Name,"face");
//
//                names.put(resultSet.getString("Name"), ma);
//                faces.add(ma);
//
//            }
//            ResultSet resultSet = statement.executeQuery(allFaces);
//            connection.close();
//
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }

    public static Mat processFace(Net net, Mat img) {
        Mat blob = Dnn.blobFromImage(img, 1. / 255, new Size(96, 96), Scalar.all(0), true, false);
        net.setInput(blob);
        return net.forward().clone();
    }

    public static boolean recognize(Mat a, ArrayList faces) {   //can be used with later use with mysql
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Net net = Dnn.readNetFromTorch("openface.nn4.small2.v1.t7");
        //   Dnn.
        //   Mat img1 = Imgcodecs.imread("../img/face1.png");
        //Mat img2 = Imgcodecs.imread(path);
        Iterator it = faces.iterator();
        while (it.hasNext()) {


            Mat feature1 = processFace(net, (Mat)it.next());
            Mat feature2 = processFace(net, a);
            double distance = Core.norm(feature1, feature2);
            return(distance < 0.6);

        }
        return false;
    }

}