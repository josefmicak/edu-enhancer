using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;

namespace TAO_Enhancer
{
    public partial class ItemForm : Form
    {
        string identifier = "";
        List<string> correctChoiceArray = new List<string>();
        List<string> correctAnswerArray = new List<string>();
        int questionType = 0;
        public ItemForm(string id)
        {
            InitializeComponent();
            identifier = id;
            LoadItemInfo();
        }

        public void LoadItemInfo()
        {
            IdentifierLabel.Text = "Identifikátor: " + identifier;

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\items\\" + identifier + "\\qti.xml");
            while (xmlReader.Read())
            {
                //if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                //Debug.WriteLine(xmlReader.Name);
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (xmlReader.Name == "assessmentItem")
                    {
                        TitleLabel.Text = "Nadpis: " + xmlReader.GetAttribute("title");
                        LabelLabel.Text = "Označení: " + xmlReader.GetAttribute("label");
                    }
                    else if (xmlReader.Name == "responseDeclaration")
                    {
                        if (xmlReader.GetAttribute("cardinality") == "ordered" && xmlReader.GetAttribute("baseType") == "identifier")
                        {
                            questionType = 1;//Typ otázky = seřazení pojmů
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "identifier")
                        {
                            questionType = 2;//Typ otázky = více odpovědí (abc); více odpovědí může být správně
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "pair")
                        {
                            questionType = 3;//Typ otázky = spojování párů
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                        {
                            questionType = 4;//Typ otázky = více otázek (tabulka); více odpovědí může být správně
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                        {
                            questionType = 5;//Typ otázky = volná odpověď
                        }
                        else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                        {
                            questionType = 6;//Typ otázky = výběr z více možností (dropdown), jen jedna odpověď je správně
                        }
                    }
                }
                switch (questionType)
                {
                    case 1:
                        if (xmlReader.Name == "prompt")
                        {
                            QuestionLabel.Text = "Otázka: " + xmlReader.ReadElementContentAsString();
                        }

                        if (xmlReader.Name == "value")
                        {
                            string value = xmlReader.ReadElementContentAsString();
                            if (value.Length > 1)//TODO: Později předělat kód, ať to zjišťuje jestli jsme v correctResponse (správně) nebo v defaultValue (špatně)
                            {
                                correctChoiceArray.Add(value);
                            }
                        }

                        if(xmlReader.Name == "simpleChoice")
                        {
                            foreach (string answer in correctChoiceArray)
                            {
                                if(xmlReader.GetAttribute("identifier") == answer)
                                {
                                    correctAnswerArray.Add(xmlReader.ReadElementContentAsString());
                                }
                            }
                        }

                        string correctAnswer = "";
                        foreach (string answer in correctAnswerArray)
                        {
                            correctAnswer += answer + "\n";
                        }
                        CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;

                        break;
                }
            }

        }
    }
}
