import simplenlg.framework.*;
import simplenlg.lexicon.*;
import simplenlg.realiser.english.*;
import simplenlg.phrasespec.*;
import simplenlg.morphology.english.*;
import simplenlg.features.*;

public class changePOS {
    public static void main(String[] args) {
			Lexicon lexicon = Lexicon.getDefaultLexicon();
			NLGFactory nlgFactory = new NLGFactory(lexicon);
			Realiser realiser = new Realiser(lexicon);
      //XMLLexicon lex = new XMLLexicon("../../../simplenlg-v4.4.3/src/simplenlg/lexicon/default-lexicon.xml");



      //String to use for error detection
      String error_str = "!!ERROR!!";

      String verb = args[0].toLowerCase();;
      String targetPOS = args[1].toLowerCase();;
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
                      System.out.println(error_str);
                    }
                    else{
                      System.out.println(baseform);
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
                      System.out.println(first);
                    }
                    else if(first.equals(second) && !second.equals(third)){
                      System.out.println(first+"!!!"+third);
                    }
                    else if(!first.equals(second) && second.equals(third)){
                      System.out.println(first+"!!!"+second);
                    }
                    else if(first.equals(third) && !second.equals(third)) {
                      System.out.println(first+"!!!"+second);
                    }
                    else {
                      System.out.println(first+"!!!"+second+"!!!"+third);
                    }
                    break;
        case "vbg": output = lexicon.getWordFromVariant(verb, LexicalCategory.VERB).getFeatureAsString(LexicalFeature.PRESENT_PARTICIPLE);
                    if(output==null) {
                      System.out.println(error_str);
                    }
                    else{
                      System.out.println(output);
                    }

                    break;
        case "vbn": output = lexicon.getWordFromVariant(verb, LexicalCategory.VERB).getFeatureAsString(LexicalFeature.PAST_PARTICIPLE);
                    if(output==null) {
                      System.out.println(error_str);
                    }
                    else{
                      System.out.println(output);
                    }
                    break;
        case "vbp": infl.setFeature(Feature.TENSE, Tense.PRESENT);
                    infl.setFeature(Feature.PERSON, Person.FIRST);
                    first = realiser.realise(infl).getRealisation();
                    infl.setFeature(Feature.PERSON, Person.SECOND);
                    second = realiser.realise(infl).getRealisation();
                    if(!first.equals(second)){
                      System.out.println(first+"!!!"+second);
                    }
                    else{
                      System.out.println(second);
                    }
                    break;
        case "vbz": infl.setFeature(Feature.TENSE, Tense.PRESENT);
                    infl.setFeature(Feature.PERSON, Person.THIRD);
                    output = realiser.realise(infl).getRealisation();
                    System.out.println(output);
                    break;
        default: System.out.println(error_str);
                 return;
      }

    }
}
