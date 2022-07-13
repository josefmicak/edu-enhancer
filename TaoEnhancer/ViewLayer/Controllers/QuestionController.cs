using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;
using System.Text.RegularExpressions;

namespace ViewLayer.Controllers
{
    public class QuestionController : Controller
    {
        /// <summary>
        /// Returns the list of questions with certain parameters - name/number identifiers, and test part/section they belong to from the test.xml file
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test</param>
        /// <param name="testNumberIdentifier">Number identifier of the selected test</param>
        /// <returns>the list of questions and their name/number identifiers, and test part/section they belong to</returns>
        public List<(string, string, string, string)> LoadQuestionParameters(string testNameIdentifier, string testNumberIdentifier)
        {
            List<(string, string, string, string)> questionParameters = new List<(string, string, string, string)>();
            string testPart = "";
            string testSection = "";
            string questionNameIdentifier;
            string questionNumberIdentifier;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testPart")
                {
                    testPart = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentSection")
                {
                    testSection = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    questionNameIdentifier = xmlReader.GetAttribute("identifier");
                    string questionNumberIdentifierToSplit = xmlReader.GetAttribute("href");
                    string[] questionNumberIdentifierSplitBySlash = questionNumberIdentifierToSplit.Split(@"/");
                    questionNumberIdentifier = questionNumberIdentifierSplitBySlash[3];
                    questionParameters.Add((testPart, testSection, questionNameIdentifier, questionNumberIdentifier));
                }
            }

            return questionParameters;
        }

        /// <summary>
        /// Returns the list of questions with all their parameters
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the selected test</param>
        /// <param name="testNumberIdentifier">Number identifier of the selected test</param>
        /// <returns>the list of questions with all their parameters</returns>
        public List<QuestionTemplate> LoadQuestionTemplates(string testNameIdentifier, string testNumberIdentifier)
        {
            int i = 0;

            //this list won't be actually returned because its elements may not be in the correct order
            List<QuestionTemplate> questionTemplatesTemp = new List<QuestionTemplate>();

            //a separate function LoadQuestionParameters is used here because some of the question parameters are located in the test.xml file, while others are in the qti.xml file
            List<(string, string, string, string)> questionParameters = LoadQuestionParameters(testNameIdentifier, testNumberIdentifier);

            if (Directory.Exists(Settings.GetTestItemsPath(testNameIdentifier)))
            {
                foreach (var directory in Directory.GetDirectories(Settings.GetTestItemsPath(testNameIdentifier)))
                {
                    foreach (var file in Directory.GetFiles(directory))
                    {
                        string[] fileSplitBySlash = file.Split(Settings.GetPathSeparator());
                        if (fileSplitBySlash[fileSplitBySlash.Length - 1] != "qti.xml")
                        {
                            continue;
                        }
                        else
                        {
                            XmlReader xmlReader = XmlReader.Create(file);
                            while (xmlReader.Read())
                            {
                                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                                {
                                    if (xmlReader.Name == "assessmentItem")
                                    {
                                        for (int j = 0; j < questionParameters.Count; j++)
                                        {
                                            if (questionParameters[j].Item4 == xmlReader.GetAttribute("identifier"))
                                            {
                                                //(int questionPoints, bool questionPointsDetermined) = GetQuestionPoints(testNameIdentifier, itemNumberIdenfifier);
                                                //questionTemplatesTemp.Add((itemNumberIdenfifier, itemNameIdenfifier, title, label, questionPoints, questionPointsDetermined));
                                                QuestionTemplate questionTemplate = new QuestionTemplate();
                                                questionTemplate.QuestionNameIdentifier = questionParameters[j].Item3;
                                                questionTemplate.QuestionNumberIdentifier = questionParameters[j].Item4;
                                                questionTemplate.Title = xmlReader.GetAttribute("title");
                                                questionTemplate.Label = xmlReader.GetAttribute(" label");
                                                questionTemplatesTemp.Add(questionTemplate);
                                                i++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else { throw Exceptions.TestItemsPathNotFoundException; }

            //correction of potential wrong order of elements in the questionTemplatesTemp array
            List<QuestionTemplate> questionTemplates = new List<QuestionTemplate>();
            for (int k = 0; k < questionParameters.Count; k++)
            {
                for (int l = 0; l < questionTemplatesTemp.Count; l++)
                {
                    if (questionParameters[k].Item4 == questionTemplatesTemp[l].QuestionNumberIdentifier)
                    {
                        questionTemplates.Add(questionTemplatesTemp[l]);
                    }
                }
            }

            return questionTemplates;
        }

        /// <summary>
        /// Returns the selected question template
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="testNumberIdentifier">Number identifier of the test that the selected question belongs to</param>
        /// <param name="questionNameIdentifier">Name identifier of the selected question</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the selected question template</returns>
        public QuestionTemplate LoadQuestionTemplate(string testNameIdentifier, string testNumberIdentifier, string questionNameIdentifier, string questionNumberIdentifier)
        {
            List<QuestionTemplate> questionTemplatesList = LoadQuestionTemplates(testNameIdentifier, testNumberIdentifier);

            for (int i = 0; i < questionTemplatesList.Count; i++)
            {
                QuestionTemplate questionTemplate = questionTemplatesList[i];
                if (questionNameIdentifier == questionTemplate.QuestionNameIdentifier && questionNumberIdentifier == questionTemplate.QuestionNumberIdentifier)
                {
                    questionTemplate.SubquestionTemplateList = LoadSubquestionTemplates(testNameIdentifier, questionNumberIdentifier);
                    return questionTemplate;
                }
            }
            return null;
        }

        /// <summary>
        /// Returns the list of all subquestion templates that are included in the selected question
        /// </summary>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the list of all subquestion templates (by question template)</returns>
        public List<SubquestionTemplate> LoadSubquestionTemplates(string testNameIdentifier, string questionNumberIdentifier)
        {
            List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
                    string subquestionIdentifier = xmlReader.GetAttribute("identifier");
                    subquestionTemplate.SubquestionIdentifier = subquestionIdentifier;

                    int subquestionType = GetSubquestionType(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier);
                    subquestionTemplate.SubquestionType = subquestionType;

                    subquestionTemplate.ImageSource = GetSubquestionImage(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier);

                    subquestionTemplate.PossibleAnswerList = GetPossibleAnswerList(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier, subquestionType);

                    subquestionTemplate.CorrectAnswerList = GetCorrectAnswerList(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier, subquestionType);

                    subquestionTemplate.SubquestionText = GetSubquestionText(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier, subquestionType, subquestionTemplate.CorrectAnswerList.Count);

                    subquestionTemplates.Add(subquestionTemplate);
                }
            }
            return subquestionTemplates;
        }

        public SubquestionTemplate LoadSubquestionTemplate(string testNameIdentifier, string questionNumberIdentifier, string selectedSubquestionIdentifier)
        {
            SubquestionTemplate subquestionTemplate = new SubquestionTemplate();
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                string subquestionIdentifier = xmlReader.GetAttribute("identifier");
                if(selectedSubquestionIdentifier == subquestionIdentifier)
                {
                    subquestionTemplate.SubquestionIdentifier = subquestionIdentifier;

                    int subquestionType = GetSubquestionType(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier);
                    subquestionTemplate.SubquestionType = subquestionType;

                    subquestionTemplate.ImageSource = GetSubquestionImage(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier);

                    subquestionTemplate.PossibleAnswerList = GetPossibleAnswerList(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier, subquestionType);

                    subquestionTemplate.CorrectAnswerList = GetCorrectAnswerList(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier, subquestionType);

                    subquestionTemplate.SubquestionText = GetSubquestionText(subquestionIdentifier, testNameIdentifier, questionNumberIdentifier, subquestionType, subquestionTemplate.CorrectAnswerList.Count);
                }
            }
            return subquestionTemplate;
        }

        /// <summary>
        /// Returns the type of subquestion
        /// </summary>
        /// <param name="selectedSubquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the type of subquestion</returns>
        public int GetSubquestionType(string selectedSubquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier)
        {
            int subquestionType = 0;
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            bool singleCorrectAnswer = false;//questionType = 6 nebo 7; jediná správná odpověď

            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    string subquestionIdentifier = xmlReader.GetAttribute("identifier");
                    
                    //skip other subquestions
                    if (subquestionIdentifier != null && subquestionIdentifier != selectedSubquestionIdentifier)
                    {
                        xmlReader.Skip();
                    }

                    if (xmlReader.GetAttribute("cardinality") == "ordered" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        subquestionType = 1;//Typ otázky = seřazení pojmů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        subquestionType = 2;//Typ otázky = více odpovědí (abc); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "pair")
                    {
                        subquestionType = 3;//Typ otázky = spojování párů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                    {
                        subquestionType = 4;//Typ otázky = více otázek (tabulka); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                    {
                        subquestionType = 5;//Typ otázky = volná odpověď; odpověď není předem daná
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "integer")
                    {
                        subquestionType = 10;//Typ otázky = volná odpověď; odpověď není předem daná
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        singleCorrectAnswer = true;
                    }
                }

                //skip other subquestions - outside of response declaration
                var name = xmlReader.Name;
                if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                    name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                {
                    string subquestionIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    if (subquestionIdentifier != null && subquestionIdentifier != selectedSubquestionIdentifier)
                    {
                        xmlReader.Skip();
                    }
                }

                if (xmlReader.Name == "gapMatchInteraction")
                {
                    string subquestionIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    if (subquestionIdentifier == selectedSubquestionIdentifier)
                    {
                        subquestionType = 9;
                    }
                }

                if (singleCorrectAnswer)
                {
                    if (xmlReader.Name == "simpleChoice")
                    {
                        subquestionType = 6;//Typ otázky = výběr z více možností (abc), jen jedna odpověď je správně
                    }
                }

                if (xmlReader.Name == "textEntryInteraction" && subquestionType == 5)
                {
                    subquestionType = 8;//Typ otázky = volná odpověď; odpověď je předem daná
                }
            }

            if (singleCorrectAnswer && subquestionType == 0)
            {
                subquestionType = 7;//Typ otázky = výběr z více možností (dropdown), jen jedna odpověď je správně
            }
            return subquestionType;
        }

        public string GetSubquestionTypeText(int subquestionType)
        {
            switch (subquestionType)
            {
                case 0:
                    return "Neznámý nebo nepodporovaný typ otázky!";
                case 1:
                    return "Seřazení pojmů";
                case 2:
                    return "Výběr z více možností; více možných správných odpovědí";
                case 3:
                    return "Spojování pojmů";
                case 4:
                    return "Více otázek k jednomu pojmu; více možných správných odpovědí";
                case 5:
                    return "Volná odpověď, správná odpověď není automaticky určena";
                case 6:
                    return "Výběr z více možností; jedna správná odpověd";
                case 7:
                    return "Výběr z více možností (doplnění textu); jedna správná odpověď";
                case 8:
                    return "Volná odpověď, správná odpověď je automaticky určena";
                case 9:
                    return "Dosazování pojmů do mezer";
                case 10:
                    return "Posuvník; jedna správná odpověď (číslo)";
                default:
                    return "Neznámý nebo nepodporovaný typ otázky!";
            }
        }

        public string[] SubquestionTypeTextArray = { 
        "Neznámý nebo nepodporovaný typ otázky!",
        "Seřazení pojmů",
        "Výběr z více možností; více možných správných odpovědí",
        "Spojování pojmů",
        "Více otázek k jednomu pojmu; více možných správných odpovědí",
        "Volná odpověď, správná odpověď není automaticky určena",
        "Výběr z více možností; jedna správná odpověd",
        "Výběr z více možností (doplnění textu); jedna správná odpověď",
        "Volná odpověď, správná odpověď je automaticky určena",
        "Dosazování pojmů do mezer",
        "Posuvník; jedna správná odpověď (číslo)"};

        /// <summary>
        /// Returns the subquestion text
        /// </summary>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <param name="subquestionType">Type of the selected subquestion</param>
        /// <returns>the text of subquestion</returns>
        public string GetSubquestionText(string subquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier, int subquestionType, int correctAnswersCount)
        {
            string subquestionText = "";
            string subquestionTextTemporary = "";
            int amountOfGaps = 0;//for subquestionType 9 (multiple gaps)
            int simpleMatchSetCounter = 0;//for subquestionType 4
            bool pTagVisited = false;
            bool divTagVisited = false;
            bool identifierCheck = false;//checks whether the current subquestion's identifier matches the "subquestionIdentifier" variable
            int nodeCount = 0;//counts how many nodes have been read by the XmlReader
            int oldNodeCount = 0;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                nodeCount++;
                if(xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                        {
                            identifierCheck = false;
                            xmlReader.Skip();
                        }
                        else
                        {
                            identifierCheck = true;
                        }
                    }
                }

                //after the node with the subquestion's identifier has been read, it's necessary to set identifierCheck to false
                if (xmlReader.NodeType == XmlNodeType.EndElement)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction")
                    {
                        identifierCheck = false;
                    }
                }

                //in subquestion types 7 and 8 it's impossible to read their identifiers before reading (at least a part) of their text
                if (subquestionType == 7 || subquestionType == 8 || subquestionType == 9)
                {
                    if(xmlReader.Name == "p" && xmlReader.NodeType == XmlNodeType.Element)
                    {
                        pTagVisited = true;
                        if(subquestionType == 7 || subquestionType == 8)
                        {
                            subquestionTextTemporary = xmlReader.ReadString() + "(DOPLŇTE)";
                            oldNodeCount = nodeCount;
                        }
                        else
                        {
                            if (identifierCheck)
                            {
                                subquestionText += xmlReader.ReadString();
                            }
                        }
                    }

                    //the identifier that has been read matches the identifier of the selected subquestion 
                    if ((subquestionType == 7 || subquestionType == 8) && nodeCount == oldNodeCount && xmlReader.GetAttribute("responseIdentifier") == subquestionIdentifier)
                    {
                        identifierCheck = true;
                    }

                    if(identifierCheck)
                    {
                        if (pTagVisited)
                        {
                            if (subquestionType == 7 || subquestionType == 8)
                            {
                                subquestionText = subquestionTextTemporary;

                                //there may be more text following the gap - this text is added to current subquestion text
                                subquestionText += xmlReader.ReadString();
                            }
                            else
                            {
                                amountOfGaps++;
                                subquestionText += xmlReader.ReadString();
                                if (amountOfGaps <= correctAnswersCount)
                                {
                                    subquestionText += "(DOPLŇTE[" + amountOfGaps + "])";
                                }
                            }
                        }

                        //stops adding text to subquestion text after </p> tag is reached
                        if (pTagVisited && xmlReader.Name == "p")
                        {
                            pTagVisited = false;
                        }
                    }
                }
                else
                {
                    if(xmlReader.Name == "prompt")
                    {
                        //using ReadSubtree ensures that every part of subquestion text is properly read and added
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            while (innerReader.Read())
                            {
                                subquestionText += innerReader.ReadString();
                            }
                        }
                    }

                    //in this type of subquestion it is necessary to add text located in some of the simpleAssociableChoice tags to the subquestion text
                    if (subquestionType == 4)
                    {
                        if (xmlReader.Name == "simpleMatchSet")
                        {
                            simpleMatchSetCounter++;
                        }

                        if(simpleMatchSetCounter == 2 && xmlReader.Name == "simpleMatchSet")
                        {
                            subquestionText += "(";
                        }

                        if (simpleMatchSetCounter == 3)
                        {
                            if (xmlReader.Name == "simpleAssociableChoice")
                            {
                                subquestionText += xmlReader.ReadString() + ", ";
                            }
                        }

                        if (simpleMatchSetCounter == 4 && xmlReader.Name == "simpleMatchSet")
                        {
                            subquestionText = subquestionText.Substring(0, subquestionText.Length - 2);//remove reduntant comma
                            subquestionText += ")";
                        }
                    }
                }
            }

            //subquestion text may still be empty due to the text being located in <div> tag - correction
            if (subquestionText == "")
            {
                nodeCount = 0;
                identifierCheck = false;
                xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
                while (xmlReader.Read())
                {
                    nodeCount++;
                    if (xmlReader.Name == "div" && xmlReader.GetAttribute("class") == "col-12" && xmlReader.NodeType == XmlNodeType.Element)
                    {
                        divTagVisited = true;
                        if (subquestionType == 7 || subquestionType == 8)
                        {
                            subquestionTextTemporary = xmlReader.ReadString() + "(DOPLŇTE)";
                            oldNodeCount = nodeCount;
                        }
                        else
                        {
                            if (identifierCheck)
                            {
                                subquestionText += xmlReader.ReadString();
                            }
                        }
                    }

                    if ((subquestionType == 7 || subquestionType == 8) && nodeCount == oldNodeCount && xmlReader.GetAttribute("responseIdentifier") == subquestionIdentifier)
                    {
                        identifierCheck = true;
                    }

                    if (identifierCheck)
                    {
                        if (divTagVisited)
                        {
                            if (subquestionType == 7 || subquestionType == 8)
                            {
                                subquestionText = subquestionTextTemporary;
                                subquestionText += xmlReader.ReadString();
                            }
                            else
                            {
                                amountOfGaps++;
                                subquestionText += xmlReader.ReadString();
                                if (amountOfGaps <= correctAnswersCount)
                                {
                                    subquestionText += "(DOPLŇTE[" + amountOfGaps + "])";
                                }
                            }
                        }

                        if (divTagVisited && xmlReader.Name == "div" && xmlReader.NodeType == XmlNodeType.EndElement)
                        {
                            break;//after the correct subquestion text has been read it's necessary to exit the loop to prevent reading further text that may not belong to this subquestion

                        }
                    }
                }
            }
            return subquestionText;
        }

        /// <summary>
        /// Returns the subquestion image source (if one exists)
        /// </summary>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the subquestion image source</returns>
        public string GetSubquestionImage(string subquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier)
        {
            string imageSource = "";

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                    }
                }

                if (xmlReader.Name == "img")
                {
                    imageSource = "https://localhost:7026/images/" + testNameIdentifier + "/items/" + questionNumberIdentifier  + "/" + xmlReader.GetAttribute("src");
                    return imageSource;
                }
            }

            return imageSource;
        }

        /// <summary>
        /// Returns the list of possible answers of the selected subquestion
        /// </summary>
        /// <param name="subquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// /// <param name="subquestionType">Type of the selected subquestion</param>
        /// <returns>the list of possible answers</returns>
        public List<string> GetPossibleAnswerList(string subquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier, int subquestionType)
        {
            List<string> possibleAnswerList = new List<string>();
            int simpleMatchSetCounter = 0;//for subquestionType 4

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                var name = xmlReader.Name;
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != subquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                        else
                        {
                            if (name == "sliderInteraction")
                            {
                                string lowerBound = xmlReader.GetAttribute("lowerBound");
                                string upperBound = xmlReader.GetAttribute("upperBound");
                                possibleAnswerList.Add(lowerBound + " - " + upperBound);
                            }
                        }
                    }
                }

                if(subquestionType != 4)
                {
                    if (name == "simpleAssociableChoice" || name == "gapText" || name == "simpleChoice" || name == "inlineChoice")
                    {
                        string possibleAnswer = xmlReader.ReadString();
                        possibleAnswerList.Add(possibleAnswer);
                    }
                }
                else
                {
                    //in case the subquestion type is 4, not every option located within a simpleAssociableChoice can be added (some have already been added to the subquestion text)
                    if (name == "simpleMatchSet")
                    {
                        simpleMatchSetCounter++;
                    }

                    if(name == "simpleAssociableChoice" && simpleMatchSetCounter == 1)
                    {
                        string possibleAnswer = xmlReader.ReadString();
                        possibleAnswerList.Add(possibleAnswer);
                    }
                }
            }

            return possibleAnswerList;
        }

        /// <summary>
        /// Returns the list of answer identifiers (tuples that contain the identifier of the answer and the text of the answer)
        /// </summary>
        /// <param name="selectedSubquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <returns>the list of answer identifiers</returns>
        public List<(string, string)> GetAnswerIdentifiers(string selectedSubquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier)
        {
            List<(string, string)> answerIdentifiers = new List<(string, string)>();

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                var name = xmlReader.Name;
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" ||
                        name == "extendedTextInteraction" || name == "orderInteraction" || name == "associateInteraction" || name == "inlineChoiceInteraction")
                    {
                        //skip other subquestions
                        if (xmlReader.GetAttribute("responseIdentifier") != selectedSubquestionIdentifier)
                        {
                            xmlReader.Skip();
                        }
                    }

                    if (name == "simpleChoice" || name == "simpleAssociableChoice" || name == "gapText" || name == "inlineChoice")
                    {
                        string answerIdentifier = xmlReader.GetAttribute("identifier");
                        string answerText = xmlReader.ReadString();
                        answerIdentifiers.Add((answerIdentifier, answerText));
                    }
                }
            }
            return answerIdentifiers;
        }

        /// <summary>
        /// Returns the list of correct answers of the selected subquestion
        /// </summary>
        /// <param name="selectedSubquestionIdentifier">Subquestion identifier of the selected subquestion</param>
        /// <param name="testNameIdentifier">Name identifier of the test that the selected question belongs to</param>
        /// <param name="questionNumberIdentifier">Number identifier of the selected question</param>
        /// <param name="subquestionType">Type of the selected subquestion</param>
        /// <returns>the list of correct answers</returns>
        public List<string> GetCorrectAnswerList(string selectedSubquestionIdentifier, string testNameIdentifier, string questionNumberIdentifier, int subquestionType)
        {
            List<string> correctIdentifierList = new List<string>();//identifiers of correct choices
            List<string> correctAnswerList = new List<string>();//text of correct choices
            List<(string, string)> answerIdentifiers = GetAnswerIdentifiers(selectedSubquestionIdentifier, testNameIdentifier, questionNumberIdentifier);

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, questionNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    string subquestionIdentifier = xmlReader.GetAttribute("identifier");
                    if(subquestionIdentifier != selectedSubquestionIdentifier)
                    {
                        xmlReader.Skip();
                    }
                }
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.Name == "value")
                {
                    correctIdentifierList.Add(xmlReader.ReadString());
                }

                //skips the <value> tag that's not relevant to correct answers
                if(xmlReader.Name == "outcomeDeclaration")
                {
                    xmlReader.Skip();
                }
            }

            //slider - the correct answer is one number
            if (subquestionType == 10)
            {
                correctAnswerList = correctIdentifierList;
            }

            for (int i = 0; i < correctIdentifierList.Count; i++)
            {
                for(int j = 0; j < answerIdentifiers.Count; j++)
                {
                    //correct answers of these subquestion types consist of doubles (such as matching two terms)
                    if(subquestionType == 3 || subquestionType == 4)
                    {
                        string[] splitIdentifiersBySpace = correctIdentifierList[i].Split(" ");
                        if(splitIdentifiersBySpace[0] == answerIdentifiers[j].Item1)
                        {
                            if (correctAnswerList.Count <= i)
                            {
                                correctAnswerList.Add(answerIdentifiers[j].Item2);
                            }
                            else
                            {
                                correctAnswerList[i] = answerIdentifiers[j].Item2 + " -> " + correctAnswerList[i];
                            }
                        }
                        if (splitIdentifiersBySpace[1] == answerIdentifiers[j].Item1)
                        {
                            if (correctAnswerList.Count <= i)
                            {
                                correctAnswerList.Add(answerIdentifiers[j].Item2);
                            }
                            else
                            {
                                correctAnswerList[i] = answerIdentifiers[j].Item2 + " -> " + correctAnswerList[i];
                            }
                        }
                    }
                    //correct answers of this type of subquestion are entered into gaps
                    else if(subquestionType == 9)
                    {
                        string[] splitIdentifiersBySpace = correctIdentifierList[i].Split(" ");
                        if (splitIdentifiersBySpace[0] == answerIdentifiers[j].Item1)
                        {
                            correctAnswerList.Add("[" + (correctAnswerList.Count + 1) + "] - " + answerIdentifiers[j].Item2);
                        }
                    }
                    else
                    {
                        if (correctIdentifierList[i] == answerIdentifiers[j].Item1)
                        {
                            correctAnswerList.Add(answerIdentifiers[j].Item2);
                        }
                    }
                }
            }
            return correctAnswerList;
        }
    }
}
