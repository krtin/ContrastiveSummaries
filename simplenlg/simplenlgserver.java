import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;
import java.io.InputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import simplenlg.framework.*;
import simplenlg.lexicon.*;
import simplenlg.realiser.english.*;
import simplenlg.phrasespec.*;
import simplenlg.morphology.english.*;
import simplenlg.features.*;

 public class simplenlgserver {

   public static void main(String args[]) throws IOException {
     ServerSocket server = new ServerSocket(8181);
     System.out.println("Listening for connection on port 8181 ....");
     while (true) {
       try (Socket socket = server.accept()) {
         Date today = new Date();

         InputStream inputstream = socket.getInputStream();
         String response = "!!ERROR!!";
         try {
                String getstring = readInputHeaders(inputstream);
                //System.out.println(getstring);


                String [] getparams = getstring.split("&");
                if(getparams.length==2){
                  String noun="", verb="", pos="";
                  for(int i=0; i<getparams.length; i++){

                      String [] tmp = getparams[i].split("=");
                      String param = tmp[0];
                      String value = tmp[1];
                      //System.out.println(param);
                      if(param.equals("noun")){
                        noun = value;
                      }
                      else if(param.equals("verb")) {
                        verb = value;
                      }
                      else if(param.equals("pos")){
                        pos = value;
                      }
                  }
                  if((!noun.equals("") || !verb.equals("")) && !pos.equals("")){
                    if(!noun.equals("")){
                      //change plurality
                      response = changePlurality(noun, pos);
                      System.out.println("Successfully Completed Number conversion from "+noun+" to "+response+" ("+pos+")");
                    }
                    else{
                      //change verb pos tag
                      response = changePOS(verb, pos);
                      System.out.println("Successfully Completed Verb POS from "+verb+" to "+response+" ("+pos+")");
                    }
                  }
                }


            } catch (Throwable t) {
                /*do nothing*/

            }
         String httpResponse = "HTTP/1.1 200 OK\r\n\r\n" + response;
         socket.getOutputStream().write(httpResponse.getBytes("UTF-8"));
       }
     }
   }

   private static String readInputHeaders(InputStream inputstream) throws Throwable {
            BufferedReader br = new BufferedReader(new InputStreamReader(inputstream));
            String returnvalue = "!!ERROR!!";
            while(true) {
                String s = br.readLine();
                String requesttype = s.substring(0, Math.min(s.length(), 3));
                if(requesttype.toLowerCase().equals("get")){
                  String [] arrOfStr = s.split(" ");
                  if(arrOfStr.length!=3) {
                    break;
                  }
                  else{
                    returnvalue = arrOfStr[1];
                    returnvalue = returnvalue.split("\\?")[1];

                  }
                }
                //System.out.println(s);
                if(s == null || s.trim().length() == 0) {
                    break;
                }
            }

            return returnvalue;
    }

    private static String changePOS(String verb, String targetPOS) {

    			Lexicon lexicon = Lexicon.getDefaultLexicon();
    			NLGFactory nlgFactory = new NLGFactory(lexicon);
    			Realiser realiser = new Realiser(lexicon);
          //XMLLexicon lex = new XMLLexicon("../../../simplenlg-v4.4.3/src/simplenlg/lexicon/default-lexicon.xml");

          //if the verb starts with ' then return it as it is
          String firstchar = verb.substring(0, Math.min(verb.length(), 1));
          if(firstchar.equals("'")){
            return verb;
          }

          //String to use for error detection
          String error_str = "!!ERROR!!";

          verb = verb.toLowerCase();
          targetPOS = targetPOS.toLowerCase();
          String output = "";
          String first = "";
          String second = "";
          String third = "";

          //get baseform
          String baseform = lexicon.getWordFromVariant(verb, LexicalCategory.VERB).getBaseForm();
          WordElement word = lexicon.getWord(baseform, LexicalCategory.VERB);
          InflectedWordElement infl = new InflectedWordElement(word);

          switch(targetPOS) {
            case "vb": if(baseform==null || baseform.equals("")) {
                          output = error_str;
                        }
                        else{
                          output = baseform;
                        }
                        break;
            case "vbd": infl.setFeature(Feature.TENSE, Tense.PAST);
                        infl.setFeature(Feature.PERSON, Person.FIRST);
                        first = realiser.realise(infl).getRealisation();
                        infl.setFeature(Feature.PERSON, Person.SECOND);
                        second = realiser.realise(infl).getRealisation();
                        infl.setFeature(Feature.PERSON, Person.THIRD);
                        third = realiser.realise(infl).getRealisation();
                        if(first.equals(second) && second.equals(third)){
                          output = first;
                        }
                        else if(first.equals(second) && !second.equals(third)){
                          output = first+"!!!"+third;
                        }
                        else if(!first.equals(second) && second.equals(third)){
                          output = first+"!!!"+second;
                        }
                        else if(first.equals(third) && !second.equals(third)) {
                          output = first+"!!!"+second;
                        }
                        else {
                          output = first+"!!!"+second+"!!!"+third;
                        }
                        break;
            case "vbg": output = lexicon.getWordFromVariant(verb, LexicalCategory.VERB).getFeatureAsString(LexicalFeature.PRESENT_PARTICIPLE);
                        if(output==null) {
                          output = error_str;
                        }

                        break;
            case "vbn": output = lexicon.getWordFromVariant(verb, LexicalCategory.VERB).getFeatureAsString(LexicalFeature.PAST_PARTICIPLE);
                        if(output==null) {
                          output = error_str;
                        }
                        break;
            case "vbp": infl.setFeature(Feature.TENSE, Tense.PRESENT);
                        infl.setFeature(Feature.PERSON, Person.FIRST);
                        first = realiser.realise(infl).getRealisation();
                        infl.setFeature(Feature.PERSON, Person.SECOND);
                        second = realiser.realise(infl).getRealisation();
                        if(!first.equals(second)){
                          output = first+"!!!"+second;
                        }
                        else{
                          output = second;
                        }
                        break;
            case "vbz": infl.setFeature(Feature.TENSE, Tense.PRESENT);
                        infl.setFeature(Feature.PERSON, Person.THIRD);
                        output = realiser.realise(infl).getRealisation();
                        break;
            default: output = error_str;

          }
          return output;
        }

        private static String changePlurality(String noun, String targetPOS) {
        			Lexicon lexicon = Lexicon.getDefaultLexicon();
        			Realiser realiser = new Realiser(lexicon);
              //XMLLexicon lex = new XMLLexicon("../../../simplenlg-v4.4.3/src/simplenlg/lexicon/default-lexicon.xml");



              //String to use for error detection
              String error_str = "!!ERROR!!";

              noun = noun.toLowerCase();
              targetPOS = targetPOS.toLowerCase();
              String output = "";
              String first = "";
              String second = "";
              String third = "";

              //get baseform
              String baseform = lexicon.getWordFromVariant(noun, LexicalCategory.NOUN).getBaseForm();
              WordElement word = lexicon.getWord(baseform, LexicalCategory.NOUN);
              InflectedWordElement infl = new InflectedWordElement(word);

              switch(targetPOS) {
                case "nn": if(baseform==null || baseform.equals("")) {
                              output = error_str;
                            }
                            else{
                              output = baseform;
                            }
                            break;
                case "nns": infl.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
                            output = realiser.realise(infl).getRealisation();

                           break;
                default: output = error_str;

              }
              return output;


        }

 }
