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
using System.Xml;

namespace TAO_Enhancer
{
    public partial class ItemsForm : Form
    {
        int chosenItem = -1;
        public ItemsForm()
        {
            InitializeComponent();
            LoadItems();
        }

        public void LoadItems()
        {
            //Složka items - prozatím přesouvám manuálně ten imsmanifest do té složky, to bude chtít do budoucna změnit; možná je ten manifest i zbytečný?
            int gridViewRow = 0;
            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\items"))
            {
                string identifier = Path.GetFileName(directory);
                string title = "";
                string label = "";

                XmlReader xmlReader = XmlReader.Create(directory + "\\qti.xml");
                while(xmlReader.Read())
                {
                    if((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentItem"))
                    {
                        if(xmlReader.HasAttributes)
                        {
                            title = xmlReader.GetAttribute("title");
                            label = xmlReader.GetAttribute("label");
                        }                 
                    }
                }

                ItemsGridView.Rows.Add();
                ItemsGridView.Rows[gridViewRow].Cells[0].Value = identifier;
                ItemsGridView.Rows[gridViewRow].Cells[1].Value = title;
                ItemsGridView.Rows[gridViewRow].Cells[2].Value = label;
                gridViewRow++;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(chosenItem == -1)
            {
                MessageBox.Show("Chyba - nevybral jste žadnou otázku. Prosím vyberte otázku kliknutím na příslušný řádek.", "Nebyla vybrána otázka", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                new ItemForm(ItemsGridView.Rows[chosenItem].Cells[0].Value.ToString()).Show();
                Hide();
            }
        }

        private void ItemsGridView_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            chosenItem = ItemsGridView.CurrentCell.RowIndex;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new TeacherForm().Show();
            Hide();
        }
    }
}
