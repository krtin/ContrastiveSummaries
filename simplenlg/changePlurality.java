
import simplenlg.framework.InflectedWordElement;
import simplenlg.framework.LexicalCategory;
import simplenlg.framework.WordElement;
import simplenlg.lexicon.Lexicon;
import simplenlg.realiser.english.Realiser;
//import simplenlg.phrasespec.*;
//import simplenlg.morphology.english.*;
import simplenlg.features.NumberAgreement;
import simplenlg.features.Feature;


public class changePlurality {
    public static void main(String[] args) {
			Lexicon lexicon = Lexicon.getDefaultLexicon();
			Realiser realiser = new Realiser(lexicon);
      //XMLLexicon lex = new XMLLexicon("../../../simplenlg-v4.4.3/src/simplenlg/lexicon/default-lexicon.xml");



      //String to use for error detection
      String error_str = "!!ERROR!!";

      String verb = args[0].toLowerCase();
      String targetPOS = args[1].toLowerCase();
      String output = "";
      String first = "";
      String second = "";
      String third = "";

      //get baseform
      String baseform = lexicon.getWordFromVariant(verb, LexicalCategory.NOUN).getBaseForm();
      WordElement word = lexicon.getWord(baseform, LexicalCategory.NOUN);
      InflectedWordElement infl = new InflectedWordElement(word);

      switch(targetPOS) {
        case "nn": if(baseform==null || baseform.equals("")) {
                      System.out.println(error_str);
                    }
                    else{
                      System.out.println(baseform);
                    }
                    break;
        case "nns": infl.setFeature(Feature.NUMBER, NumberAgreement.PLURAL);
                    output = realiser.realise(infl).getRealisation();
                    System.out.println(output);
                   break;
        default: System.out.println(error_str);
                 return;
      }

    }
}
