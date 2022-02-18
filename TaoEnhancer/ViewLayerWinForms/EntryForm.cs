using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TAO_Enhancer
{
    public partial class EntryForm : Form
    {
        public EntryForm()
        {
            InitializeComponent();
            this.Text = "TAO Enhancer - Hlavní menu";
        }

        private void StudentButton_Click(object sender, EventArgs e)
        {
            new StudentForm().Show();
            Hide();
        }

        private void TeacherButton_Click(object sender, EventArgs e)
        {
            new TeacherForm().Show();
            Hide();
        }

        private void CloseButton_Click(object sender, EventArgs e)
        {
            Environment.Exit(0);
        }
    }
}
