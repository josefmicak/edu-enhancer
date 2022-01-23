using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using VDS.RDF;
using VDS.RDF.Parsing;
using VDS.RDF.Query;

namespace TAO_Enhancer
{
    public partial class StudentForm : Form
    {
        public StudentForm()
        {
            InitializeComponent();
            LoadTestTakers();
        }

        public void LoadTestTakers()
        {
            int gridViewRow = 0;
            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\testtakers"))
            {
                string extension = Path.GetExtension(file);
                if(extension == ".rdf")
                {
                    IGraph g = new Graph();
                    FileLoader.Load(g, file);
                    IEnumerable<INode> nodes = g.AllNodes;
                    int nodeLine = 1;//TODO 1: Velmi špatný kód - minimálně to tady chce udělat podmínky jako if(node == ns0:userFirstName), zatím ale vůbec nevím jak na to
                    string login = "", name = "", surname = "";
                    foreach (INode node in nodes)
                    {
                        if (nodeLine == 3)
                        {
                            login = node.ToString();
                        }
                        else if (nodeLine == 9)
                        {
                            name = node.ToString();
                        }
                        else if (nodeLine == 11)
                        {
                            surname = node.ToString();
                        }
                        nodeLine++;
                    }
                    TestTakersGridView.Rows.Add();
                    TestTakersGridView.Rows[gridViewRow].Cells[0].Value = login;
                    TestTakersGridView.Rows[gridViewRow].Cells[1].Value = name + " " + surname;
                    gridViewRow++;
                }
            }
/*
            IGraph g = new Graph();
            FileLoader.Load(g, "C:\\xampp\\exported\\testtakers\\john_1642833661.rdf");
            IEnumerable<INode> nodes = g.AllNodes;
            int nodeLine = 1;//Velmi špatný kód - minimálně to tady chce udělat podmínky jako if(node == ns0:userFirstName), zatím ale vůbec nevím jak na to
            string login = "", name = "", surname = "";
            foreach (INode node in nodes)
            {
                if(nodeLine == 3)
                {
                    login = node.ToString();
                }
                else if(nodeLine == 9)
                {
                    name = node.ToString();
                }
                else if(nodeLine == 11)
                {
                    surname = node.ToString();
                }
                nodeLine++;
            }
            TestTakersGridView.Rows.Add();
            TestTakersGridView.Rows[0].Cells[0].Value = login;
            TestTakersGridView.Rows[0].Cells[1].Value = name + " " + surname;*/
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new EntryForm().Show();
            Hide();
        }
    }
}
