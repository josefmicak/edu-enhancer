
namespace TAO_Enhancer
{
    partial class StudentForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.TestTakersGridView = new System.Windows.Forms.DataGridView();
            this.LoginColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.JmenoColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.label1 = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.TestTakersGridView)).BeginInit();
            this.SuspendLayout();
            // 
            // TestTakersGridView
            // 
            this.TestTakersGridView.AllowUserToAddRows = false;
            this.TestTakersGridView.AllowUserToDeleteRows = false;
            this.TestTakersGridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.TestTakersGridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.LoginColumn,
            this.JmenoColumn});
            this.TestTakersGridView.Location = new System.Drawing.Point(69, 118);
            this.TestTakersGridView.Name = "TestTakersGridView";
            this.TestTakersGridView.ReadOnly = true;
            this.TestTakersGridView.RowTemplate.Height = 25;
            this.TestTakersGridView.Size = new System.Drawing.Size(436, 150);
            this.TestTakersGridView.TabIndex = 0;
            // 
            // LoginColumn
            // 
            this.LoginColumn.HeaderText = "Login";
            this.LoginColumn.Name = "LoginColumn";
            this.LoginColumn.ReadOnly = true;
            // 
            // JmenoColumn
            // 
            this.JmenoColumn.HeaderText = "Jméno";
            this.JmenoColumn.Name = "JmenoColumn";
            this.JmenoColumn.ReadOnly = true;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.label1.Location = new System.Drawing.Point(69, 75);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(150, 20);
            this.label1.TabIndex = 1;
            this.label1.Text = "Seznam studentů";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(369, 408);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 2;
            this.button1.Text = "Zpět";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // StudentForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.TestTakersGridView);
            this.Name = "StudentForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "StudentForm";
            ((System.ComponentModel.ISupportInitialize)(this.TestTakersGridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView TestTakersGridView;
        private System.Windows.Forms.DataGridViewTextBoxColumn LoginColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn JmenoColumn;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button1;
    }
}